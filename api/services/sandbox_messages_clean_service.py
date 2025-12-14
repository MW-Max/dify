import datetime
import logging
from collections.abc import Sequence
from typing import cast

from sqlalchemy import delete, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from configs import dify_config
from enums.cloud_plan import CloudPlan
from extensions.ext_database import db
from models.account import Tenant
from models.model import (
    App,
    AppAnnotationHitHistory,
    DatasetRetrieverResource,
    Message,
    MessageAgentThought,
    MessageAnnotation,
    MessageChain,
    MessageFeedback,
    MessageFile,
)
from models.web import SavedMessage
from services.billing_service import BillingService

logger = logging.getLogger(__name__)


class SandboxMessagesCleanService:
    """
    Service for cleaning expired messages from sandbox plan tenants.
    """    
    @classmethod
    def clean_sandbox_messages(cls) -> dict[str, int]:
        """
        Clean expired messages from sandbox plan tenants.
        Core implementation following cursor-based pagination.
        
        Steps:
        1. Iterate all tenants using cursor pagination (batch of 200)
        2. Batch fetch plans from billing service
        3. Filter sandbox tenants
        4. For each batch of sandbox tenants, delete their expired messages
        5. Batch delete related records using message_id
        """
        plan_sandbox_clean_message_day = datetime.datetime.now() - datetime.timedelta(
            days=dify_config.PLAN_SANDBOX_CLEAN_MESSAGE_DAY_SETTING
        )
        
        stats = {
            "batches": 0,
            "total_messages": 0,
            "total_deleted": 0,
            "tenants_processed": 0,
        }
        
        # Step 1: Cursor-based pagination through all tenants
        last_tenant_id = None

        logger.info("clean_messages: start clean messages before %s", plan_sandbox_clean_message_day)
        
        while True:
            # Fetch tenant batch using cursor
            with Session(db.engine, expire_on_commit=False) as session:
                if last_tenant_id is None:
                    stmt = (
                        select(Tenant.id)
                        .order_by(Tenant.id)
                        .limit(dify_config.SANDBOX_MESSAGES_CLEAN_TENANT_BATCH_SIZE)
                    )
                    tenants = list(session.scalars(stmt).all())
                else:
                    stmt = (
                        select(Tenant.id)
                        .where(Tenant.id > last_tenant_id)
                        .order_by(Tenant.id)
                        .limit(dify_config.SANDBOX_MESSAGES_CLEAN_TENANT_BATCH_SIZE)
                    )
                    tenants = list(session.scalars(stmt).all())
            
            if not tenants:
                logger.info("clean_messages: no more tenants to process")
                break
            
            stats['batches'] += 1
            
            tenant_ids = tenants
            last_tenant_id = tenant_ids[-1]
            
            # Step 2: Batch fetch plans from billing API
            # This calls external billing service API, so it's intentionally placed outside
            # any db session to avoid long transactions and connection pool exhaustion
            tenant_plans = cls._batch_fetch_tenant_plans(tenant_ids)
            
            # Step 3: Filter sandbox tenants
            sandbox_tenant_ids = [
                tenant_id for tenant_id in tenant_ids
                if tenant_plans.get(tenant_id) == CloudPlan.SANDBOX
            ]
            
            if not sandbox_tenant_ids:
                logger.info("clean_messages (batch %s): no sandbox tenants, skip this batch", stats["batches"])
                continue
            
            # Get app_ids for these sandbox tenants
            with Session(db.engine, expire_on_commit=False) as session:
                stmt = select(App.id).where(App.tenant_id.in_(sandbox_tenant_ids))
                app_ids = list(session.scalars(stmt).all())
            
            if not app_ids:
                logger.info("clean_messages (batch %s): no apps, skip this batch", stats["batches"])
                continue
            
            # Step 4: Delete messages for these sandbox tenants (limit-based pagination)
            current_batch_deleted_messages = 0
            while True:
                with Session(db.engine, expire_on_commit=False) as session:
                    stmt = (
                        select(Message.id)
                        .where(
                            Message.app_id.in_(app_ids),
                            Message.created_at < plan_sandbox_clean_message_day
                        )
                        .limit(dify_config.SANDBOX_MESSAGES_CLEAN_MESSAGE_BATCH_SIZE)
                    )
                    message_ids = list(session.scalars(stmt).all())
                    
                    if not message_ids:
                        logger.info(
                            "clean_messages (batch %s): no more messages, end this batch", stats["batches"]
                        )
                        break
                    
                    stats["total_messages"] += len(message_ids)
                    
                    # Step 5: Batch delete related records
                    cls._batch_delete_message_relations(session, message_ids)
                    
                    # Delete messages
                    delete_stmt = delete(Message).where(Message.id.in_(message_ids))
                    delete_result = cast(CursorResult, session.execute(delete_stmt))
                    message_deleted = delete_result.rowcount
                    session.commit()
                    
                    current_batch_deleted_messages += message_deleted
                    stats["total_deleted"] += message_deleted
                
            stats["tenants_processed"] += len(sandbox_tenant_ids)

            logger.info(
                "clean_messages (batch %s): processed %s sandbox tenants, deleted %s messages",
                stats["batches"],
                len(sandbox_tenant_ids),
                current_batch_deleted_messages,
            )
            
        logger.info(
            "clean_messages completed: "
            "total batches: %s, "
            "total sandbox tenants: %s, "
            "total messages: %s, "
            "total messages deleted: %s",
            stats["batches"],
            stats["tenants_processed"],
            stats["total_messages"],
            stats["total_deleted"],
        )
        
        return stats
    
    @classmethod
    def _batch_fetch_tenant_plans(cls, tenant_ids: Sequence[str]) -> dict[str, str]:
        """
        Batch fetch tenant plans from billing API.
        
        Args:
            tenant_ids: List of tenant IDs
            
        Returns:
            Dict mapping tenant_id to plan
        """
        tenant_plans: dict[str, str] = {}
        
        try:
            bulk_plans = BillingService.get_plan_bulk(tenant_ids)
            if bulk_plans:
                for tenant_id, plan in bulk_plans.items():
                    if isinstance(plan, str) and plan:
                        tenant_plans[tenant_id] = plan
        except Exception as e:
            # If we unable to fetch some tenants plans, log the error and continue the batch
            logger.exception("clean_messages: failed to batch fetch plans")
        
        return tenant_plans
    
    @classmethod
    def _batch_delete_message_relations(cls, session: Session, message_ids: Sequence[str]) -> None:
        """
        Batch delete all related records for given message IDs.
        
        Args:
            session: Database session
            message_ids: List of message IDs to delete relations for
        """
        if not message_ids:
            return
        
        # Delete all related records in batch
        session.execute(
            delete(MessageFeedback).where(MessageFeedback.message_id.in_(message_ids))
        )
        
        session.execute(
            delete(MessageAnnotation).where(MessageAnnotation.message_id.in_(message_ids))
        )
        
        session.execute(
            delete(MessageChain).where(MessageChain.message_id.in_(message_ids))
        )
        
        session.execute(
            delete(MessageAgentThought).where(MessageAgentThought.message_id.in_(message_ids))
        )
        
        session.execute(
            delete(MessageFile).where(MessageFile.message_id.in_(message_ids))
        )
        
        session.execute(
            delete(SavedMessage).where(SavedMessage.message_id.in_(message_ids))
        )
        
        session.execute(
            delete(AppAnnotationHitHistory).where(AppAnnotationHitHistory.message_id.in_(message_ids))
        )
        
        session.execute(
            delete(DatasetRetrieverResource).where(DatasetRetrieverResource.message_id.in_(message_ids))
        )
