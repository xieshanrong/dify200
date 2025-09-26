class FileService:
    _session_maker: sessionmaker

    def __init__(self, session_factory: sessionmaker | Engine | None = None):
        """
        初始化文件服务类。

        Args:
            session_factory: 数据库会话工厂或引擎，可选
        """
        if isinstance(session_factory, Engine):
            # 如果传入的是引擎，创建会话工厂
            self._session_maker = sessionmaker(bind=session_factory)
        elif isinstance(session_factory, sessionmaker):
            # 如果传入的是会话工厂，直接使用
            self._session_maker = session_factory
        else:
            # 如果传入的既不是引擎也不是会话工厂，抛出异常
            raise AssertionError("must be a sessionmaker or an Engine.")

    def upload_file(
        self,
        *,
        filename: str,
        content: bytes,
        mimetype: str,
        user: Union[Account, EndUser, Any],
        source: Literal["datasets"] | None = None,
        source_url: str = "",
    ) -> UploadFile:
        """
        上传文件到存储并保存到数据库。

        Args:
            filename: 文件名
            content: 文件内容（字节）
            mimetype: 文件 MIME 类型
            user: 上传文件的用户
            source: 文件来源，可选 "datasets"
            source_url: 文件来源 URL，可选

        Returns:
            UploadFile: 上传的文件对象

        Raises:
            ValueError: 文件名包含非法字符
            UnsupportedFileTypeError: 文件类型不支持
            FileTooLargeError: 文件大小超过限制
        """
        # 获取文件扩展名
        extension = os.path.splitext(filename)[1].lstrip(".").lower()

        # 检查文件名是否包含非法字符
        if any(c in filename for c in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]):
            raise ValueError("Filename contains invalid characters")

        # 如果文件名过长，截取部分
        if len(filename) > 200:
            filename = filename.split(".")[0][:200] + "." + extension

        # 检查文件类型（仅限于数据集来源时）
        if source == "datasets" and extension not in DOCUMENT_EXTENSIONS:
            raise UnsupportedFileTypeError()

        # 获取文件大小
        file_size = len(content)

        # 检查文件大小是否超出限制
        if not FileService.is_file_size_within_limit(extension=extension, file_size=file_size):
            raise FileTooLargeError()

        # 生成文件唯一标识符
        file_uuid = str(uuid.uuid4())

        # 获取租户 ID
        current_tenant_id = extract_tenant_id(user)

        # 生成文件键
        file_key = f"upload_files/{current_tenant_id or ''}/{file_uuid}.{extension}"

        # 将文件保存到存储
        storage.save(file_key, content)

        # 创建并保存上传文件到数据库
        upload_file = UploadFile(
            tenant_id=current_tenant_id or "",
            storage_type=dify_config.STORAGE_TYPE,
            key=file_key,
            name=filename,
            size=file_size,
            extension=extension,
            mime_type=mimetype,
            created_by_role=(
                CreatorUserRole.ACCOUNT if isinstance(user, Account) else CreatorUserRole.END_USER
            ),
            created_by=user.id,
            created_at=naive_utc_now(),
            used=False,
            hash=hashlib.sha3_256(content).hexdigest(),
            source_url=source_url,
        )

        # 如果 source_url 未设置，生成签名 URL
        if not upload_file.source_url:
            upload_file.source_url = file_helpers.get_signed_file_url(upload_file_id=upload_file.id)

        # 使用会话制造商添加并提交上传文件
        with self._session_maker(expire_on_commit=False) as session:
            session.add(upload_file)
            session.commit()

        return upload_file

    @staticmethod
    def is_file_size_within_limit(*, extension: str, file_size: int) -> bool:
        """
        检查文件大小是否在允许的限制内。

        Args:
            extension: 文件扩展名
            file_size: 文件大小（字节）

        Returns:
            bool: 文件大小是否在限制内
        """
        if extension in IMAGE_EXTENSIONS:
            file_size_limit = dify_config.UPLOAD_IMAGE_FILE_SIZE_LIMIT * 1024 * 1024
        elif extension in VIDEO_EXTENSIONS:
            file_size_limit = dify_config.UPLOAD_VIDEO_FILE_SIZE_LIMIT * 1024 * 1024
        elif extension in AUDIO_EXTENSIONS:
            file_size_limit = dify_config.UPLOAD_AUDIO_FILE_SIZE_LIMIT * 1024 * 1024
        else:
            file_size_limit = dify_config.UPLOAD_FILE_SIZE_LIMIT * 1024 * 1024

        return file_size <= file_size_limit

    @staticmethod
    def upload_text(text: str, text_name: str) -> UploadFile:
        """
        上传文本内容为文件。

        Args:
            text: 文本内容
            text_name: 文件名

        Returns:
            UploadFile: 上传的文件对象
        """
        assert isinstance(current_user, Account)
        assert current_user.current_tenant_id is not None

        # 如果文件名过长，截取部分
        if len(text_name) > 200:
            text_name = text_name[:200]

        # 生成文件唯一标识符
        file_uuid = str(uuid.uuid4())

        # 生成文件键
        file_key = f"upload_files/{current_user.current_tenant_id}/{file_uuid}.txt"

        # 将文本内容保存到存储
        storage.save(file_key, text.encode("utf-8"))

        # 创建并保存上传文件到数据库
        upload_file = UploadFile(
            tenant_id=current_user.current_tenant_id,
            storage_type=dify_config.STORAGE_TYPE,
            key=file_key,
            name=text_name,
            size=len(text),
            extension="txt",
            mime_type="text/plain",
            created_by=current_user.id,
            created_by_role=CreatorUserRole.ACCOUNT,
            created_at=naive_utc_now(),
            used=True,
            used_by=current_user.id,
            used_at=naive_utc_now(),
        )

        with self._session_maker(expire_on_commit=False) as session:
            session.add(upload_file)
            session.commit()

        return upload_file

    def get_file_preview(self, file_id: str) -> str:
        """
        获取文件的文本预览。

        Args:
            file_id: 文件 ID

        Returns:
            str: 文件的文本预览内容

        Raises:
            NotFound: 文件未找到
            UnsupportedFileTypeError: 文件类型不支持
        """
        with self._session_maker(expire_on_commit=False) as session:
            upload_file = session.query(UploadFile).where(UploadFile.id == file_id).first()

        if not upload_file:
            raise NotFound("File not found")

        # 检查文件类型是否为支持的文档类型
        extension = upload_file.extension
        if extension.lower() not in DOCUMENT_EXTENSIONS:
            raise UnsupportedFileTypeError()

        # 提取文本内容并限制字数
        text = ExtractProcessor.load_from_upload_file(upload_file, return_text=True)
        text = text[:PREVIEW_WORDS_LIMIT] if text else ""

        return text

    def get_image_preview(self, file_id: str, timestamp: str, nonce: str, sign: str) -> tuple:
        """
        验证图像文件的签名并返回文件生成器。

        Args:
            file_id: 文件 ID
            timestamp: 时间戳
            nonce: 随机数
            sign: 签名

        Returns:
            tuple: 文件生成器和 MIME 类型

        Raises:
            NotFound: 文件未找到或签名无效
            UnsupportedFileTypeError: 文件类型不支持
        """
        # 验证图像文件签名
        result = file_helpers.verify_image_signature(
            upload_file_id=file_id, timestamp=timestamp, nonce=nonce, sign=sign
        )
        if not result:
            raise NotFound("File not found or signature is invalid")

        with self._session_maker(expire_on_commit=False) as session:
            upload_file = session.query(UploadFile).where(UploadFile.id == file_id).first()

        if not upload_file:
            raise NotFound("File not found or signature is invalid")

        # 检查文件类型是否为支持的图像类型
        extension = upload_file.extension
        if extension.lower() not in IMAGE_EXTENSIONS:
            raise UnsupportedFileTypeError()

        # 返回文件生成器和 MIME 类型
        generator = storage.load(upload_file.key, stream=True)
        return generator, upload_file.mime_type

    def get_file_generator_by_file_id(self, file_id: str, timestamp: str, nonce: str, sign: str) -> tuple:
        """
        验证文件签名并返回文件生成器。

        Args:
            file_id: 文件 ID
            timestamp: 时间戳
            nonce: 随机数
            sign: 签名

        Returns:
            tuple: 文件生成器和上传文件对象

        Raises:
            NotFound: 文件未找到或签名无效
        """
        # 验证文件签名
        result = file_helpers.verify_file_signature(
            upload_file_id=file_id, timestamp=timestamp, nonce=nonce, sign=sign
        )
        if not result:
            raise NotFound("File not found or signature is invalid")

        with self._session_maker(expire_on_commit=False) as session:
            upload_file = session.query(UploadFile).where(UploadFile.id == file_id).first()

        if not upload_file:
            raise NotFound("File not found or signature is invalid")

        # 返回文件生成器和上传文件对象
        generator = storage.load(upload_file.key, stream=True)
        return generator, upload_file

    def get_public_image_preview(self, file_id: str) -> tuple:
        """
        获取公共图像文件的预览。

        Args:
            file_id: 文件 ID

        Returns:
            tuple: 文件内容生成器和 MIME 类型

        Raises:
            NotFound: 文件未找到或签名无效
            UnsupportedFileTypeError: 文件类型不支持
        """
        with self._session_maker(expire_on_commit=False) as session:
            upload_file = session.query(UploadFile).where(UploadFile.id == file_id).first()

        if not upload_file:
            raise NotFound("File not found or signature is invalid")

        # 检查文件类型是否为支持的图像类型
        extension = upload_file.extension
        if extension.lower() not in IMAGE_EXTENSIONS:
            raise UnsupportedFileTypeError()

        # 返回文件内容生成器和 MIME 类型
        generator = storage.load(upload_file.key)
        return generator, upload_file.mime_type
