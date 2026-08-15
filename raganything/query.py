"""
Query functionality for RAGAnything

Contains all query-related methods for both text and multimodal queries
"""

import json
import hashlib
import re
import time
from typing import Dict, List, Any
from pathlib import Path
from lightrag import QueryParam, LightRAG
from lightrag.utils import always_get_an_event_loop
from raganything.prompt import PROMPTS
from raganything.utils import (
    get_processor_for_type,
    encode_image_to_base64,
    validate_image_file,
    query_instruct_ctx,
)


class QueryMixin:
    """QueryMixin class containing query functionality for RAGAnything"""

    async def _ensure_lightrag(self):
        """Lazy-initialize LightRAG on first query if not already set up.

        RAGAnything.__init__ only builds an empty shell; self.lightrag stays None
        until a document has been inserted via aadd_documents. This means restarting
        the server would otherwise make aquery fail even though rag_storage on disk
        already holds data. This guard loads that existing data on first query.
        """
        if self.lightrag is not None:
            return

        self.logger.info(
            "LightRAG not yet initialized — loading existing storage from working_dir: "
            f"{self.working_dir}"
        )

        if self.llm_model_func is None:
            raise ValueError("llm_model_func must be configured before running queries")

        lightrag_params = {
            "working_dir": self.working_dir,
            "llm_model_func": self.llm_model_func,
            "embedding_func": self.embedding_func,
        }
        lightrag_params.update(self.lightrag_kwargs)

        log_params = {
            k: v
            for k, v in lightrag_params.items()
            if not callable(v)
            and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
        }
        self.logger.info(
            f"Initializing LightRAG (lazy, for query) with parameters: {log_params}"
        )

        try:
            self.lightrag = LightRAG(**lightrag_params)
            await self.lightrag.initialize_storages()
            from lightrag.kg.shared_storage import initialize_pipeline_status

            await initialize_pipeline_status()
            self.logger.info("LightRAG initialized lazily from existing storage")
        except Exception as e:
            raise ValueError(
                f"Failed to initialize LightRAG for query: {e}. "
                f"Ensure rag_storage at {self.working_dir} is valid."
            ) from e

    def _generate_multimodal_cache_key(
        self, query: str, multimodal_content: List[Dict[str, Any]], mode: str, **kwargs
    ) -> str:
        """
        Generate cache key for multimodal query

        Args:
            query: Base query text
            multimodal_content: List of multimodal content
            mode: Query mode
            **kwargs: Additional parameters

        Returns:
            str: Cache key hash
        """
        # Create a normalized representation of the query parameters
        cache_data = {
            "query": query.strip(),
            "mode": mode,
        }

        # Normalize multimodal content for stable caching
        normalized_content = []
        if multimodal_content:
            for item in multimodal_content:
                if isinstance(item, dict):
                    normalized_item = {}
                    for key, value in item.items():
                        # For file paths, use basename to make cache more portable
                        if key in [
                            "img_path",
                            "image_path",
                            "file_path",
                        ] and isinstance(value, str):
                            normalized_item[key] = Path(value).name
                        # For large content, create a hash instead of storing directly
                        elif (
                            key in ["table_data", "table_body"]
                            and isinstance(value, str)
                            and len(value) > 200
                        ):
                            normalized_item[f"{key}_hash"] = hashlib.md5(
                                value.encode()
                            ).hexdigest()
                        else:
                            normalized_item[key] = value
                    normalized_content.append(normalized_item)
                else:
                    normalized_content.append(item)

        cache_data["multimodal_content"] = normalized_content

        # Add relevant kwargs to cache data
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "stream",
                "response_type",
                "top_k",
                "max_tokens",
                "temperature",
                "system_prompt",
                # "only_need_context",
                # "only_need_prompt",
            ]
        }
        cache_data.update(relevant_kwargs)

        # Generate hash from the cache data
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()

        return f"multimodal_query:{cache_hash}"

    async def aquery(
        self, query: str, mode: str = "mix", system_prompt: str | None = None, **kwargs
    ) -> str:
        """
        Pure text query - directly calls LightRAG's query functionality

        Args:
            query: Query text
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            system_prompt: Optional system prompt to include.
            **kwargs: Other query parameters, will be passed to QueryParam
                - vlm_enhanced: bool, default True when vision_model_func is available.
                  If True, will parse image paths in retrieved context and replace them
                  with base64 encoded images for VLM processing.

        Returns:
            str: Query result
        """
        await self._ensure_lightrag()

        # 用 LLM 生成 instruct，优化查询 embedding 质量
        instruct = await self._generate_instruct(query)
        token = query_instruct_ctx.set(instruct)
        try:
            return await self._aquery_inner(
                query, mode=mode, system_prompt=system_prompt, **kwargs
            )
        finally:
            query_instruct_ctx.reset(token)

    async def _aquery_inner(
        self, query: str, mode: str = "mix", system_prompt: str | None = None, **kwargs
    ) -> str:
        """aquery 内部实现，在 instruct 上下文保护下执行"""
        # Check if VLM enhanced query should be used
        vlm_enhanced = kwargs.pop("vlm_enhanced", None)

        # Auto-determine VLM enhanced based on availability
        if vlm_enhanced is None:
            vlm_enhanced = (
                hasattr(self, "vision_model_func")
                and self.vision_model_func is not None
            )

        # Use VLM enhanced query if enabled and available
        if (
            vlm_enhanced
            and hasattr(self, "vision_model_func")
            and self.vision_model_func
        ):
            return await self.aquery_vlm_enhanced(
                query, mode=mode, system_prompt=system_prompt, **kwargs
            )
        elif vlm_enhanced and (
            not hasattr(self, "vision_model_func") or not self.vision_model_func
        ):
            self.logger.warning(
                "VLM enhanced query requested but vision_model_func is not available, falling back to normal query"
            )

        callback_manager = getattr(self, "callback_manager", None)
        query_start_time = time.time()

        if callback_manager is not None:
            callback_manager.dispatch(
                "on_query_start",
                query=query,
                mode=mode,
            )

        # Create query parameters
        query_param = QueryParam(mode=mode, **kwargs)

        self.logger.info(f"Executing text query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        try:
            # Call LightRAG's query method
            result = await self.lightrag.aquery(
                query, param=query_param, system_prompt=system_prompt
            )
        except Exception as exc:
            if callback_manager is not None:
                callback_manager.dispatch(
                    "on_query_error",
                    query=query,
                    mode=mode,
                    error=exc,
                )
            raise

        self.logger.info("Text query completed")
        if callback_manager is not None:
            duration = time.time() - query_start_time
            result_len = len(result) if isinstance(result, str) else 0
            callback_manager.dispatch(
                "on_query_complete",
                query=query,
                mode=mode,
                duration_seconds=duration,
                result_length=result_len,
            )
        return result

    async def _generate_instruct(self, query: str) -> str | None:
        """
        使用 LLM 从用户查询中生成 instruct（任务指令），用于优化 embedding 检索质量。

        Instruct 告诉 embedding 模型当前检索任务的类型，让向量更针对性。
        例如：查询"机器学习论文" -> instruct="Given a research paper query, retrieve relevant research paper"

        如果 LLM 不可用或生成失败，返回 None（使用默认 behavior）。
        """
        llm = getattr(self, "llm_model_func", None)
        if not llm:
            return None

        try:
            prompt = PROMPTS["INSTRUCT_GENERATION"].format(query=query.strip())
            result = await llm(prompt, system_prompt="You are a helpful assistant.")
            instruct = result.strip().strip('"\'')
            if instruct:
                self.logger.info(
                    f"Generated instruct for query [{query[:60]}...]: {instruct}"
                )
                return instruct
        except Exception as e:
            self.logger.warning(f"Failed to generate instruct: {e}")
        return None

    async def aquery_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]] = None,
        mode: str = "mix",
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """
        Multimodal query - combines text and multimodal content for querying

        Args:
            query: Base query text
            multimodal_content: List of multimodal content, each element contains:
                - type: Content type ("image", "table", "equation", etc.)
                - Other fields depend on type (e.g., img_path, table_data, latex, etc.)
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            system_prompt: Optional system prompt to include in the query
            **kwargs: Other query parameters, will be passed to QueryParam

        Returns:
            str: Query result

        Examples:
            # Pure text query
            result = await rag.query_with_multimodal("What is machine learning?")

            # Image query
            result = await rag.query_with_multimodal(
                "Analyze the content in this image",
                multimodal_content=[{
                    "type": "image",
                    "img_path": "./image.jpg"
                }]
            )

            # Table query
            result = await rag.query_with_multimodal(
                "Analyze the data trends in this table",
                multimodal_content=[{
                    "type": "table",
                    "table_data": "Name,Age\nAlice,25\nBob,30"
                }]
            )
        """
        # Ensure LightRAG is initialized
        init_result = await self._ensure_lightrag_initialized()
        if not init_result or not init_result.get("success"):
            raise RuntimeError(
                f"LightRAG initialization failed: {(init_result or {}).get('error', 'unknown error')}"
            )

        self.logger.info(f"Executing multimodal query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        # If no multimodal content, fallback to pure text query
        if not multimodal_content:
            self.logger.info("No multimodal content provided, executing text query")
            return await self.aquery(
                query, mode=mode, system_prompt=system_prompt, **kwargs
            )

        # Generate cache key for multimodal query
        cache_key = self._generate_multimodal_cache_key(
            query,
            multimodal_content,
            mode,
            system_prompt=system_prompt,
            **kwargs,
        )

        # Check cache if available and enabled
        cached_result = None
        if (
            hasattr(self, "lightrag")
            and self.lightrag
            and hasattr(self.lightrag, "llm_response_cache")
            and self.lightrag.llm_response_cache
        ):
            if self.lightrag.llm_response_cache.global_config.get(
                "enable_llm_cache", True
            ):
                try:
                    cached_result = await self.lightrag.llm_response_cache.get_by_id(
                        cache_key
                    )
                    if cached_result and isinstance(cached_result, dict):
                        result_content = cached_result.get("return")
                        if result_content:
                            self.logger.info(
                                f"Multimodal query cache hit: {cache_key[:16]}..."
                            )
                            return result_content
                except Exception as e:
                    self.logger.debug(f"Error accessing multimodal query cache: {e}")

        # Process multimodal content to generate enhanced query text
        enhanced_query = await self._process_multimodal_query_content(
            query, multimodal_content
        )

        self.logger.info(
            f"Generated enhanced query length: {len(enhanced_query)} characters"
        )

        # Execute enhanced query
        result = await self.aquery(
            enhanced_query, mode=mode, system_prompt=system_prompt, **kwargs
        )

        # Save to cache if available and enabled
        if (
            hasattr(self, "lightrag")
            and self.lightrag
            and hasattr(self.lightrag, "llm_response_cache")
            and self.lightrag.llm_response_cache
        ):
            if self.lightrag.llm_response_cache.global_config.get(
                "enable_llm_cache", True
            ):
                try:
                    # Create cache entry for multimodal query
                    cache_entry = {
                        "return": result,
                        "cache_type": "multimodal_query",
                        "original_query": query,
                        "multimodal_content_count": len(multimodal_content),
                        "mode": mode,
                    }

                    await self.lightrag.llm_response_cache.upsert(
                        {cache_key: cache_entry}
                    )
                    self.logger.info(
                        f"Saved multimodal query result to cache: {cache_key[:16]}..."
                    )
                except Exception as e:
                    self.logger.debug(f"Error saving multimodal query to cache: {e}")

        # Ensure cache is persisted to disk
        if (
            hasattr(self, "lightrag")
            and self.lightrag
            and hasattr(self.lightrag, "llm_response_cache")
            and self.lightrag.llm_response_cache
        ):
            try:
                await self.lightrag.llm_response_cache.index_done_callback()
            except Exception as e:
                self.logger.debug(f"Error persisting multimodal query cache: {e}")

        self.logger.info("Multimodal query completed")
        return result

    async def aquery_vlm_enhanced(
        self,
        query: str,
        mode: str = "mix",
        system_prompt: str | None = None,
        extra_safe_dirs: List[str] = None,
        **kwargs,
    ) -> str:
        """
        VLM enhanced query - replaces image paths in retrieved context with base64 encoded images for VLM processing

        Args:
            query: User query
            mode: Underlying LightRAG query mode
            system_prompt: Optional system prompt to include
            extra_safe_dirs: Optional list of additional safe directories to allow images from
            **kwargs: Other query parameters

        Returns:
            str: VLM query result
        """
        # Ensure VLM is available
        if not hasattr(self, "vision_model_func") or not self.vision_model_func:
            raise ValueError(
                "VLM enhanced query requires vision_model_func. "
                "Please provide a vision model function when initializing RAGAnything."
            )

        # Ensure LightRAG is initialized
        init_result = await self._ensure_lightrag_initialized()
        if not init_result or not init_result.get("success"):
            raise RuntimeError(
                f"LightRAG initialization failed: {(init_result or {}).get('error', 'unknown error')}"
            )

        self.logger.info(f"Executing VLM enhanced query: {query[:100]}...")

        # Clear previous image cache
        if hasattr(self, "_current_images_base64"):
            delattr(self, "_current_images_base64")

        # 1. Get original retrieval prompt (without generating final answer)
        query_param = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
        raw_prompt = await self.lightrag.aquery(query, param=query_param)

        self.logger.debug("Retrieved raw prompt from LightRAG")

        # 2. Extract and process image paths
        enhanced_prompt, images_found = await self._process_image_paths_for_vlm(
            raw_prompt, extra_safe_dirs=extra_safe_dirs
        )

        if not images_found:
            self.logger.info("No valid images found, falling back to normal query")
            # Fallback to normal query
            query_param = QueryParam(mode=mode, **kwargs)
            return await self.lightrag.aquery(
                query, param=query_param, system_prompt=system_prompt
            )

        self.logger.info(f"Processed {images_found} images for VLM")

        # 3. Build VLM message format
        messages = self._build_vlm_messages_with_images(
            enhanced_prompt, query, system_prompt
        )

        # 4. Call VLM for question answering
        result = await self._call_vlm_with_multimodal_content(messages)

        self.logger.info("VLM enhanced query completed")
        return result

    async def aquery_with_user_images(
        self,
        query: str,
        user_image_paths: List[str] = None,
        mode: str = "mix",
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """
        User image query - combines user-uploaded images with knowledge base
        retrieval for comprehensive VLM answering.

        Flow:
        1. LightRAG retrieves relevant context from knowledge base (text only)
        2. Scans context for Image Path: markers, encodes KB images to base64
        3. Encodes user-provided images to base64
        4. Builds multimodal messages: context + KB images + user images + question
        5. Calls vision_model_func (qwen3.7-flash) for multimodal answering

        Args:
            query: User query text
            user_image_paths: List of paths to user-uploaded images
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            system_prompt: Optional system prompt
            **kwargs: Other query parameters, passed to QueryParam

        Returns:
            str: Query result
        """
        # Ensure VLM is available
        if not hasattr(self, "vision_model_func") or not self.vision_model_func:
            raise ValueError(
                "aquery_with_user_images requires vision_model_func. "
                "Please provide a vision model function when initializing RAGAnything."
            )

        # Ensure LightRAG is initialized
        init_result = await self._ensure_lightrag_initialized()
        if not init_result or not init_result.get("success"):
            raise RuntimeError(
                f"LightRAG initialization failed: {(init_result or {}).get('error', 'unknown error')}"
            )

        self.logger.info(f"Executing user image query: {query[:100]}...")

        # Clear previous image cache
        if hasattr(self, "_current_images_base64"):
            delattr(self, "_current_images_base64")

        # 1. Retrieve knowledge base context (without generating final answer)
        query_param = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
        raw_prompt = await self.lightrag.aquery(query, param=query_param)

        self.logger.debug("Retrieved raw prompt from LightRAG")

        # 2. Process KB image paths in the retrieved context
        enhanced_prompt, kb_image_count = await self._process_image_paths_for_vlm(
            raw_prompt
        )

        # 3. Encode user-provided images
        user_images_base64 = []
        if user_image_paths:
            for img_path in user_image_paths:
                try:
                    b64 = encode_image_to_base64(img_path)
                    if b64:
                        user_images_base64.append(b64)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to encode user image {img_path}: {e}"
                    )

        # If no images at all, fall back to normal query
        if not user_images_base64 and kb_image_count == 0:
            self.logger.info("No images found, falling back to normal query")
            query_param = QueryParam(mode=mode, **kwargs)
            return await self.lightrag.aquery(
                query, param=query_param, system_prompt=system_prompt
            )

        # 4. Append user images to the image list and add markers
        kb_images = getattr(self, "_current_images_base64", [])
        all_images = kb_images + user_images_base64
        self._current_images_base64 = all_images

        # Add user image markers to the prompt
        if user_images_base64:
            for i in range(len(user_images_base64)):
                marker_idx = kb_image_count + i + 1
                enhanced_prompt += (
                    f"\n[User Uploaded Image {marker_idx}]\n[VLM_IMAGE_{marker_idx}]"
                )

        self.logger.info(
            f"Total images for VLM: {len(all_images)} "
            f"(KB: {kb_image_count}, User: {len(user_images_base64)})"
        )

        # 5. Build multimodal messages and call VLM
        messages = self._build_vlm_messages_with_images(
            enhanced_prompt, query, system_prompt
        )
        result = await self._call_vlm_with_multimodal_content(messages)

        self.logger.info("User image query completed")
        return result

    async def _process_multimodal_query_content(
        self, base_query: str, multimodal_content: List[Dict[str, Any]]
    ) -> str:
        """
        Process multimodal query content to generate enhanced query text

        Args:
            base_query: Base query text
            multimodal_content: List of multimodal content

        Returns:
            str: Enhanced query text
        """
        self.logger.info("Starting multimodal query content processing...")

        enhanced_parts = [f"User query: {base_query}"]

        for i, content in enumerate(multimodal_content):
            content_type = content.get("type", "unknown")
            self.logger.info(
                f"Processing {i + 1}/{len(multimodal_content)} multimodal content: {content_type}"
            )

            try:
                # Get appropriate processor
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # Generate content description
                    description = await self._generate_query_content_description(
                        processor, content, content_type
                    )
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {description}"
                    )
                else:
                    # If no appropriate processor, use basic description
                    basic_desc = str(content)[:200]
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {basic_desc}"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                # Continue processing other content
                continue

        enhanced_query = "\n".join(enhanced_parts)
        enhanced_query += PROMPTS["QUERY_ENHANCEMENT_SUFFIX"]

        self.logger.info("Multimodal query content processing completed")
        return enhanced_query

    async def _generate_query_content_description(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """
        Generate content description for query

        Args:
            processor: Multimodal processor
            content: Content data
            content_type: Content type

        Returns:
            str: Content description
        """
        try:
            if content_type == "image":
                return await self._describe_image_for_query(processor, content)
            elif content_type == "table":
                return await self._describe_table_for_query(processor, content)
            elif content_type == "equation":
                return await self._describe_equation_for_query(processor, content)
            else:
                return await self._describe_generic_for_query(
                    processor, content, content_type
                )

        except Exception as e:
            self.logger.error(f"Error generating {content_type} description: {str(e)}")
            return f"{content_type} content: {str(content)[:100]}"

    async def _describe_image_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate image description for query"""
        image_path = content.get("img_path")
        captions = content.get("image_caption", content.get("img_caption", []))
        footnotes = content.get("image_footnote", content.get("img_footnote", []))

        if image_path and Path(image_path).exists():
            # If image exists, use vision model to generate description
            image_base64 = processor._encode_image_to_base64(image_path)
            if image_base64:
                prompt = PROMPTS["QUERY_IMAGE_DESCRIPTION"]
                description = await processor.modal_caption_func(
                    prompt,
                    image_data=image_base64,
                    system_prompt=PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"],
                )
                return description

        # If image doesn't exist or processing failed, use existing information
        parts = []
        if image_path:
            parts.append(f"Image path: {image_path}")
        if captions:
            parts.append(f"Image captions: {', '.join(captions)}")
        if footnotes:
            parts.append(f"Image footnotes: {', '.join(footnotes)}")

        return "; ".join(parts) if parts else "Image content information incomplete"

    async def _describe_table_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate table description for query"""
        table_data = content.get("table_data", "")
        table_caption = content.get("table_caption", "")

        prompt = PROMPTS["QUERY_TABLE_ANALYSIS"].format(
            table_data=table_data, table_caption=table_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"]
        )

        return description

    async def _describe_equation_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate equation description for query"""
        latex = content.get("latex", "")
        equation_caption = content.get("equation_caption", "")

        prompt = PROMPTS["QUERY_EQUATION_ANALYSIS"].format(
            latex=latex, equation_caption=equation_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"]
        )

        return description

    async def _describe_generic_for_query(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """Generate generic content description for query"""
        content_str = str(content)

        prompt = PROMPTS["QUERY_GENERIC_ANALYSIS"].format(
            content_type=content_type, content_str=content_str
        )

        description = await processor.modal_caption_func(
            prompt,
            system_prompt=PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"].format(
                content_type=content_type
            ),
        )

        return description

    async def _process_image_paths_for_vlm(
        self, prompt: str, extra_safe_dirs: List[str] = None
    ) -> tuple[str, int]:
        """
        Process image paths in prompt, keeping original paths and adding VLM markers

        Args:
            prompt: Original prompt
            extra_safe_dirs: Optional list of additional safe directories

        Returns:
            tuple: (processed prompt, image count)
        """
        enhanced_prompt = prompt
        images_processed = 0

        # Initialize image cache
        self._current_images_base64 = []

        # Enhanced regex pattern for matching image paths
        # Matches only the path ending with image file extensions
        image_path_pattern = (
            r"Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))"
        )

        # First, let's see what matches we find
        matches = re.findall(image_path_pattern, prompt)
        self.logger.info(f"Found {len(matches)} image path matches in prompt")

        def replace_image_path(match):
            nonlocal images_processed

            image_path = match.group(1).strip()
            self.logger.debug(f"Processing image path: '{image_path}'")

            # Validate path format (basic check)
            if not image_path or len(image_path) < 3:
                self.logger.warning(f"Invalid image path format: {image_path}")
                return match.group(0)  # Keep original

            # Use utility function to validate image file
            is_valid = validate_image_file(image_path)

            # Security check: only allow images from the workspace or output directories
            # to prevent indirect prompt injection from reading arbitrary system files.
            if is_valid:
                abs_image_path = Path(image_path).resolve()
                # Check if it's in the current working directory or subdirectories
                try:
                    is_in_cwd = abs_image_path.is_relative_to(Path.cwd())
                except ValueError:
                    is_in_cwd = False

                # If a config is available, check against working_dir and parser_output_dir
                is_in_safe_dir = is_in_cwd
                if hasattr(self, "config") and self.config:
                    try:
                        is_in_working = abs_image_path.is_relative_to(
                            Path(self.config.working_dir).resolve()
                        )
                        is_in_output = abs_image_path.is_relative_to(
                            Path(self.config.parser_output_dir).resolve()
                        )
                        is_in_safe_dir = is_in_safe_dir or is_in_working or is_in_output
                    except Exception:
                        pass

                # Check against extra safe directories if provided
                if not is_in_safe_dir and extra_safe_dirs:
                    for safe_dir in extra_safe_dirs:
                        try:
                            if abs_image_path.is_relative_to(Path(safe_dir).resolve()):
                                is_in_safe_dir = True
                                break
                        except Exception:
                            continue

                if not is_in_safe_dir:
                    self.logger.warning(
                        f"Blocking image path outside safe directories: {image_path}"
                    )
                    is_valid = False

            if not is_valid:
                self.logger.warning(
                    f"Image validation failed or path unsafe for: {image_path}"
                )
                return match.group(0)  # Keep original if validation fails

            try:
                # Encode image to base64 using utility function
                self.logger.debug(f"Attempting to encode image: {image_path}")
                image_base64 = encode_image_to_base64(image_path)
                if image_base64:
                    images_processed += 1
                    # Save base64 to instance variable for later use
                    self._current_images_base64.append(image_base64)

                    # Keep original path info and add VLM marker
                    result = f"Image Path: {image_path}\n[VLM_IMAGE_{images_processed}]"
                    self.logger.debug(
                        f"Successfully processed image {images_processed}: {image_path}"
                    )
                    return result
                else:
                    self.logger.error(f"Failed to encode image: {image_path}")
                    return match.group(0)  # Keep original if encoding failed

            except Exception as e:
                self.logger.error(f"Failed to process image {image_path}: {e}")
                return match.group(0)  # Keep original

        # Execute replacement
        enhanced_prompt = re.sub(
            image_path_pattern, replace_image_path, enhanced_prompt
        )

        return enhanced_prompt, images_processed

    def _build_vlm_messages_with_images(
        self, enhanced_prompt: str, user_query: str, system_prompt: str
    ) -> List[Dict]:
        """
        Build VLM message format, using markers to correspond images with text positions

        Args:
            enhanced_prompt: Enhanced prompt with image markers
            user_query: User query

        Returns:
            List[Dict]: VLM message format
        """
        images_base64 = getattr(self, "_current_images_base64", [])

        if not images_base64:
            # Pure text mode
            return [
                {
                    "role": "user",
                    "content": f"Context:\n{enhanced_prompt}\n\nUser Question: {user_query}",
                }
            ]

        # Build multimodal content
        content_parts = []

        # Split text at image markers and insert images
        text_parts = enhanced_prompt.split("[VLM_IMAGE_")

        for i, text_part in enumerate(text_parts):
            if i == 0:
                # First text part
                if text_part.strip():
                    content_parts.append({"type": "text", "text": text_part})
            else:
                # Find marker number and insert corresponding image
                marker_match = re.match(r"(\d+)\](.*)", text_part, re.DOTALL)
                if marker_match:
                    image_num = (
                        int(marker_match.group(1)) - 1
                    )  # Convert to 0-based index
                    remaining_text = marker_match.group(2)

                    # Insert corresponding image
                    if 0 <= image_num < len(images_base64):
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images_base64[image_num]}"
                                },
                            }
                        )

                    # Insert remaining text
                    if remaining_text.strip():
                        content_parts.append({"type": "text", "text": remaining_text})

        # Add user question
        content_parts.append(
            {
                "type": "text",
                "text": f"\n\nUser Question: {user_query}\n\nPlease answer based on the context and images provided.",
            }
        )
        base_system_prompt = "You are a helpful assistant that can analyze both text and image content to provide comprehensive answers."

        if system_prompt:
            full_system_prompt = base_system_prompt + " " + system_prompt
        else:
            full_system_prompt = base_system_prompt

        return [
            {
                "role": "system",
                "content": full_system_prompt,
            },
            {
                "role": "user",
                "content": content_parts,
            },
        ]

    async def _call_vlm_with_multimodal_content(self, messages: List[Dict]) -> str:
        """
        Call VLM to process multimodal content

        Args:
            messages: VLM message format

        Returns:
            str: VLM response result
        """
        try:
            user_message = messages[1]
            content = user_message["content"]
            system_prompt = messages[0]["content"]

            if isinstance(content, str):
                # Pure text mode
                result = await self.vision_model_func(
                    content, system_prompt=system_prompt
                )
            else:
                # Multimodal mode - pass complete messages directly to VLM
                result = await self.vision_model_func(
                    "",  # Empty prompt since we're using messages format
                    messages=messages,
                )

            return result

        except Exception as e:
            self.logger.error(f"VLM call failed: {e}")
            raise

    # Synchronous versions of query methods
    def query(self, query: str, mode: str = "mix", **kwargs) -> str:
        """
        Synchronous version of pure text query

        Args:
            query: Query text
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters, will be passed to QueryParam
                - vlm_enhanced: bool, default True when vision_model_func is available.
                  If True, will parse image paths in retrieved context and replace them
                  with base64 encoded images for VLM processing.

        Returns:
            str: Query result
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, mode=mode, **kwargs))

    def query_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]] = None,
        mode: str = "mix",
        **kwargs,
    ) -> str:
        """
        Synchronous version of multimodal query

        Args:
            query: Base query text
            multimodal_content: List of multimodal content, each element contains:
                - type: Content type ("image", "table", "equation", etc.)
                - Other fields depend on type (e.g., img_path, table_data, latex, etc.)
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters, will be passed to QueryParam

        Returns:
            str: Query result
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_multimodal(query, multimodal_content, mode=mode, **kwargs)
        )
