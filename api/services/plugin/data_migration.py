import json
import logging

import click
import sqlalchemy as sa

from extensions.ext_database import db
from models.provider_ids import GenericProviderID, ModelProviderID, ToolProviderID

logger = logging.getLogger(__name__)


class PluginDataMigration:
    """
    插件数据迁移类，用于处理数据库中插件相关数据的迁移工作
    """

    @classmethod
    def migrate(cls):
        """
        执行所有数据迁移操作的主方法
        按顺序迁移各种表中的插件数据
        """
        cls.migrate_db_records("providers", "provider_name", ModelProviderID)  # large table
        cls.migrate_db_records("provider_models", "provider_name", ModelProviderID)
        cls.migrate_db_records("provider_orders", "provider_name", ModelProviderID)
        cls.migrate_db_records("tenant_default_models", "provider_name", ModelProviderID)
        cls.migrate_db_records("tenant_preferred_model_providers", "provider_name", ModelProviderID)
        cls.migrate_db_records("provider_model_settings", "provider_name", ModelProviderID)
        cls.migrate_db_records("load_balancing_model_configs", "provider_name", ModelProviderID)
        cls.migrate_datasets()
        cls.migrate_db_records("embeddings", "provider_name", ModelProviderID)  # large table
        cls.migrate_db_records("dataset_collection_bindings", "provider_name", ModelProviderID)
        cls.migrate_db_records("tool_builtin_providers", "provider", ToolProviderID)

    @classmethod
    def migrate_datasets(cls):
        """
        迁移数据集表中的插件数据
        特别处理embedding_model_provider字段和retrieval_model中的reranking_provider_name字段
        """
        table_name = "datasets"
        provider_column_name = "embedding_model_provider"

        click.echo(click.style(f"Migrating [{table_name}] data for plugin", fg="white"))

        processed_count = 0
        failed_ids = []
        while True:
            # 查询需要迁移的数据集记录
            sql = f"""select id, {provider_column_name} as provider_name, retrieval_model from {table_name}
where {provider_column_name} not like '%/%' and {provider_column_name} is not null and {provider_column_name} != ''
limit 1000"""
            with db.engine.begin() as conn:
                rs = conn.execute(sa.text(sql))

                current_iter_count = 0
                for i in rs:
                    record_id = str(i.id)
                    provider_name = str(i.provider_name)
                    retrieval_model = i.retrieval_model
                    print(type(retrieval_model))

                    if record_id in failed_ids:
                        continue

                    # 检查是否需要更新retrieval_model中的reranking_provider_name字段
                    retrieval_model_changed = False
                    if retrieval_model:
                        if (
                            "reranking_model" in retrieval_model
                            and "reranking_provider_name" in retrieval_model["reranking_model"]
                            and retrieval_model["reranking_model"]["reranking_provider_name"]
                            and "/" not in retrieval_model["reranking_model"]["reranking_provider_name"]
                        ):
                            click.echo(
                                click.style(
                                    f"[{processed_count}] Migrating {table_name} {record_id} "
                                    f"(reranking_provider_name: "
                                    f"{retrieval_model['reranking_model']['reranking_provider_name']})",
                                    fg="white",
                                )
                            )
                            # 将简单的provider名称更新为完整的格式，如google更新为langgenius/gemini/google等
                            retrieval_model["reranking_model"]["reranking_provider_name"] = ModelProviderID(
                                retrieval_model["reranking_model"]["reranking_provider_name"]
                            ).to_string()
                            retrieval_model_changed = True

                    click.echo(
                        click.style(
                            f"[{processed_count}] Migrating [{table_name}] {record_id} ({provider_name})",
                            fg="white",
                        )
                    )

                    try:
                        # 准备更新参数
                        params = {"record_id": record_id}
                        update_retrieval_model_sql = ""
                        if retrieval_model and retrieval_model_changed:
                            update_retrieval_model_sql = ", retrieval_model = :retrieval_model"
                            params["retrieval_model"] = json.dumps(retrieval_model)

                        # 将provider名称更新为完整格式
                        params["provider_name"] = ModelProviderID(provider_name).to_string()

                        # 执行更新操作
                        sql = f"""update {table_name}
                        set {provider_column_name} =
                        :provider_name
                        {update_retrieval_model_sql}
                        where id = :record_id"""
                        conn.execute(sa.text(sql), params)
                        click.echo(
                            click.style(
                                f"[{processed_count}] Migrated [{table_name}] {record_id} ({provider_name})",
                                fg="green",
                            )
                        )
                    except Exception:
                        failed_ids.append(record_id)
                        click.echo(
                            click.style(
                                f"[{processed_count}] Failed to migrate [{table_name}] {record_id} ({provider_name})",
                                fg="red",
                            )
                        )
                        logger.exception(
                            "[%s] Failed to migrate [%s] %s (%s)", processed_count, table_name, record_id, provider_name
                        )
                        continue

                    current_iter_count += 1
                    processed_count += 1

            # 如果当前迭代没有处理任何记录，则退出循环
            if not current_iter_count:
                break

        click.echo(
            click.style(f"Migrate [{table_name}] data for plugin completed, total: {processed_count}", fg="green")
        )

    @classmethod
    def migrate_db_records(cls, table_name: str, provider_column_name: str, provider_cls: type[GenericProviderID]):
        """
        迁移数据库记录中的插件provider信息

        Args:
            table_name (str): 需要迁移的表名
            provider_column_name (str): provider字段名
            provider_cls (type[GenericProviderID]): provider ID类，用于转换provider名称格式
        """
        click.echo(click.style(f"Migrating [{table_name}] data for plugin", fg="white"))

        processed_count = 0
        failed_ids = []
        last_id = "00000000-0000-0000-0000-000000000000"

        # 分批处理数据，避免一次性加载过多数据
        while True:
            # 查询需要迁移的记录
            sql = f"""
                SELECT id, {provider_column_name} AS provider_name
                FROM {table_name}
                WHERE {provider_column_name} NOT LIKE '%/%'
                    AND {provider_column_name} IS NOT NULL
                    AND {provider_column_name} != ''
                    AND id > :last_id
                ORDER BY id ASC
                LIMIT 5000
            """
            params = {"last_id": last_id or ""}

            with db.engine.begin() as conn:
                rs = conn.execute(sa.text(sql), params)

                current_iter_count = 0
                batch_updates = []

                # 处理查询结果
                for i in rs:
                    current_iter_count += 1
                    processed_count += 1
                    record_id = str(i.id)
                    last_id = record_id
                    provider_name = str(i.provider_name)

                    if record_id in failed_ids:
                        continue

                    click.echo(
                        click.style(
                            f"[{processed_count}] Migrating [{table_name}] {record_id} ({provider_name})",
                            fg="white",
                        )
                    )

                    try:
                        # 将简单的provider名称更新为完整格式，如jina更新为langgenius/jina_tool/jina等
                        updated_value = provider_cls(provider_name).to_string()
                        batch_updates.append((updated_value, record_id))
                    except Exception:
                        failed_ids.append(record_id)
                        click.echo(
                            click.style(
                                f"[{processed_count}] Failed to migrate [{table_name}] {record_id} ({provider_name})",
                                fg="red",
                            )
                        )
                        logger.exception(
                            "[%s] Failed to migrate [%s] %s (%s)", processed_count, table_name, record_id, provider_name
                        )
                        continue

                # 批量更新记录
                if batch_updates:
                    update_sql = f"""
                        UPDATE {table_name}
                        SET {provider_column_name} = :updated_value
                        WHERE id = :record_id
                    """
                    conn.execute(sa.text(update_sql), [{"updated_value": u, "record_id": r} for u, r in batch_updates])
                    click.echo(
                        click.style(
                            f"[{processed_count}] Batch migrated [{len(batch_updates)}] records from [{table_name}]",
                            fg="green",
                        )
                    )

            # 如果当前迭代没有处理任何记录，则退出循环
            if not current_iter_count:
                break

        click.echo(
            click.style(f"Migrate [{table_name}] data for plugin completed, total: {processed_count}", fg="green")
        )
