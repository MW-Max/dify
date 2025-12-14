"""
Integration tests for SandboxMessagesCleanService using testcontainers.

This module provides comprehensive integration tests for the sandbox message cleanup service
using TestContainers infrastructure with real PostgreSQL database.
"""

import datetime
import uuid
from unittest.mock import patch

import pytest
from faker import Faker

from configs import dify_config
from enums.cloud_plan import CloudPlan
from extensions.ext_database import db
from models.account import Account, Tenant, TenantAccountJoin, TenantAccountRole
from models.model import (
    App,
    AppAnnotationHitHistory,
    Conversation,
    DatasetRetrieverResource,
    Message,
    MessageAgentThought,
    MessageAnnotation,
    MessageChain,
    MessageFeedback,
    MessageFile,
)
from models.web import SavedMessage
from services.sandbox_messages_clean_service import SandboxMessagesCleanService


class TestSandboxMessagesCleanService:
    """Integration tests for SandboxMessagesCleanService using testcontainers."""

    @pytest.fixture(autouse=True)
    def cleanup_database(self, db_session_with_containers):
        """Clean up database before each test to ensure isolation."""
        # Clear all test data in correct order (respecting foreign key constraints)
        db.session.query(DatasetRetrieverResource).delete()
        db.session.query(AppAnnotationHitHistory).delete()
        db.session.query(SavedMessage).delete()
        db.session.query(MessageFile).delete()
        db.session.query(MessageAgentThought).delete()
        db.session.query(MessageChain).delete()
        db.session.query(MessageAnnotation).delete()
        db.session.query(MessageFeedback).delete()
        db.session.query(Message).delete()
        db.session.query(Conversation).delete()
        db.session.query(App).delete()
        db.session.query(TenantAccountJoin).delete()
        db.session.query(Tenant).delete()
        db.session.query(Account).delete()
        db.session.commit()

    @pytest.fixture
    def mock_billing_service(self):
        """Mock BillingService to avoid external API calls."""
        with patch("services.sandbox_messages_clean_service.BillingService") as mock:
            yield mock

    @pytest.fixture
    def expired_date(self):
        """Get the expiration date for messages."""
        return datetime.datetime.now() - datetime.timedelta(days=dify_config.PLAN_SANDBOX_CLEAN_MESSAGE_DAY_SETTING)

    @pytest.fixture
    def non_expired_date(self):
        """Get a non-expired date for messages."""
        return datetime.datetime.now() - datetime.timedelta(days=1)

    def _create_account_and_tenant(self, plan="sandbox"):
        """Helper to create account and tenant."""
        fake = Faker()

        account = Account(
            email=fake.email(),
            name=fake.name(),
            interface_language="en-US",
            status="active",
        )
        db.session.add(account)
        db.session.commit()

        tenant = Tenant(
            name=fake.company(),
            plan=plan,
            status="normal",
        )
        db.session.add(tenant)
        db.session.commit()

        tenant_account_join = TenantAccountJoin(
            tenant_id=tenant.id,
            account_id=account.id,
            role=TenantAccountRole.OWNER,
        )
        db.session.add(tenant_account_join)
        db.session.commit()

        return account, tenant

    def _create_app(self, tenant, account):
        """Helper to create an app."""
        fake = Faker()

        app = App(
            tenant_id=tenant.id,
            name=fake.company(),
            description="Test app",
            mode="chat",
            enable_site=True,
            enable_api=True,
            api_rpm=60,
            api_rph=3600,
            is_demo=False,
            is_public=False,
            created_by=account.id,
            updated_by=account.id,
        )
        db.session.add(app)
        db.session.commit()

        return app

    def _create_conversation(self, app, account):
        """Helper to create a conversation."""
        conversation = Conversation(
            app_id=app.id,
            app_model_config_id=str(uuid.uuid4()),
            model_provider="openai",
            model_id="gpt-3.5-turbo",
            mode="chat",
            name="Test conversation",
            inputs={},  # Required JSON field
            status="normal",
            from_source="api",
            from_end_user_id=str(uuid.uuid4()),
        )
        db.session.add(conversation)
        db.session.commit()

        return conversation

    def _create_message(self, app, conversation, created_at=None):
        """Helper to create a message with all related records."""
        if created_at is None:
            created_at = datetime.datetime.now()

        from decimal import Decimal

        message = Message(
            app_id=app.id,
            conversation_id=conversation.id,
            model_provider="openai",
            model_id="gpt-3.5-turbo",
            inputs={},  # Required JSON field
            query="Test query",
            answer="Test answer",
            message=[{"role": "user", "text": "Test message"}],  # Required JSON field
            message_tokens=10,
            message_unit_price=Decimal("0.001"),
            answer_tokens=20,
            answer_unit_price=Decimal("0.002"),
            total_price=Decimal("0.003"),
            currency="USD",
            from_source="api",
            from_account_id=conversation.from_end_user_id,
            created_at=created_at,
        )
        db.session.add(message)
        db.session.commit()

        # Create related records
        self._create_message_relations(message)

        return message

    def _create_message_relations(self, message):
        """Helper to create all message-related records."""
        import json

        # MessageFeedback
        feedback = MessageFeedback(
            app_id=message.app_id,
            conversation_id=message.conversation_id,
            message_id=message.id,
            rating="like",
            from_source="api",
            from_end_user_id=str(uuid.uuid4()),
        )
        db.session.add(feedback)

        # MessageAnnotation
        annotation = MessageAnnotation(
            app_id=message.app_id,
            conversation_id=message.conversation_id,
            message_id=message.id,
            question="Test question",
            content="Test annotation",
            account_id=message.from_account_id,
        )
        db.session.add(annotation)

        # MessageChain
        chain = MessageChain(
            message_id=message.id,
            type="system",
            input=json.dumps({"test": "input"}),
            output=json.dumps({"test": "output"}),
        )
        db.session.add(chain)
        db.session.flush()  # Get chain.id

        # MessageAgentThought
        thought = MessageAgentThought(
            message_id=message.id,
            message_chain_id=chain.id,
            thought="Test thought",
            tool="test_tool",
            tool_input="test input",
            observation="test observation",
            position=1,
            message_files="[]",
            created_by_role="end_user",
            created_by=str(uuid.uuid4()),
        )
        db.session.add(thought)

        # MessageFile
        file = MessageFile(
            message_id=message.id,
            type="image",
            transfer_method="local_file",
            url="http://example.com/test.jpg",
            belongs_to="user",
            created_by_role="end_user",
            created_by=str(uuid.uuid4()),
        )
        db.session.add(file)

        # SavedMessage
        saved = SavedMessage(
            app_id=message.app_id,
            message_id=message.id,
            created_by_role="end_user",
            created_by=str(uuid.uuid4()),
        )
        db.session.add(saved)

        db.session.flush()  # Get annotation.id

        # AppAnnotationHitHistory
        hit = AppAnnotationHitHistory(
            app_id=message.app_id,
            annotation_id=annotation.id,
            message_id=message.id,
            source="annotation",
            question="Test question",
            account_id=message.from_account_id,
            annotation_question="Test annotation question",
            annotation_content="Test annotation content",
        )
        db.session.add(hit)

        # DatasetRetrieverResource
        resource = DatasetRetrieverResource(
            message_id=message.id,
            position=1,
            dataset_id=str(uuid.uuid4()),
            dataset_name="Test dataset",
            document_id=str(uuid.uuid4()),
            document_name="Test document",
            data_source_type="upload_file",
            segment_id=str(uuid.uuid4()),
            score=0.9,
            content="Test content",
            hit_count=1,
            word_count=10,
            segment_position=1,
            index_node_hash="test_hash",
            retriever_from="dataset",
            created_by=message.from_account_id,
        )
        db.session.add(resource)

        db.session.commit()

    def test_clean_no_tenants(self, db_session_with_containers, mock_billing_service):
        """Test cleaning when there are no tenants."""
        # Execute
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert
        assert stats["batches"] == 0
        assert stats["tenants_processed"] == 0
        assert stats["total_messages"] == 0
        assert stats["total_deleted"] == 0

        # BillingService should not be called
        mock_billing_service.get_plan_bulk.assert_not_called()

    def test_clean_no_sandbox_tenants(self, db_session_with_containers, mock_billing_service):
        """Test cleaning when there are only non-sandbox tenants."""
        # Setup: Create 3 professional tenants
        for i in range(3):
            account, tenant = self._create_account_and_tenant(plan="professional")
            app = self._create_app(tenant, account)
            conv = self._create_conversation(app, account)
            self._create_message(app, conv, created_at=datetime.datetime.now() - datetime.timedelta(days=100))

        # Mock billing service to return professional plans
        mock_billing_service.get_plan_bulk.return_value = {
            tenant.id: CloudPlan.PROFESSIONAL for tenant in db.session.query(Tenant).all()
        }

        # Execute
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert
        assert stats["batches"] == 1
        assert stats["tenants_processed"] == 0
        assert stats["total_messages"] == 0
        assert stats["total_deleted"] == 0

        # All messages should still exist
        assert db.session.query(Message).count() == 3

    def test_clean_sandbox_tenants_no_expired_messages(
        self, db_session_with_containers, mock_billing_service, non_expired_date
    ):
        """Test cleaning sandbox tenants with no expired messages."""
        # Setup: Create 2 sandbox tenants with recent messages
        tenant_ids = []
        for i in range(2):
            account, tenant = self._create_account_and_tenant(plan="sandbox")
            tenant_ids.append(tenant.id)
            app = self._create_app(tenant, account)
            conv = self._create_conversation(app, account)
            self._create_message(app, conv, created_at=non_expired_date)

        # Mock billing service
        mock_billing_service.get_plan_bulk.return_value = dict.fromkeys(tenant_ids, CloudPlan.SANDBOX)

        # Execute
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert
        assert stats["batches"] == 1
        assert stats["tenants_processed"] == 2
        assert stats["total_messages"] == 0
        assert stats["total_deleted"] == 0

        # All messages should still exist
        assert db.session.query(Message).count() == 2

    def test_clean_sandbox_tenants_with_expired_messages(
        self, db_session_with_containers, mock_billing_service, expired_date
    ):
        """Test cleaning sandbox tenants with expired messages."""
        # Setup: Create 2 sandbox tenants with expired messages
        tenant_ids = []
        for i in range(2):
            account, tenant = self._create_account_and_tenant(plan="sandbox")
            tenant_ids.append(tenant.id)
            app = self._create_app(tenant, account)
            conv = self._create_conversation(app, account)
            # Create 3 expired messages per tenant
            for j in range(3):
                self._create_message(app, conv, created_at=expired_date - datetime.timedelta(days=j))

        # Mock billing service
        mock_billing_service.get_plan_bulk.return_value = dict.fromkeys(tenant_ids, CloudPlan.SANDBOX)

        # Execute
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert
        assert stats["batches"] == 1
        assert stats["tenants_processed"] == 2
        assert stats["total_messages"] == 6  # 2 tenants * 3 messages
        assert stats["total_deleted"] == 6

        # All messages should be deleted
        assert db.session.query(Message).count() == 0

        # All related records should be deleted
        assert db.session.query(MessageFeedback).count() == 0
        assert db.session.query(MessageAnnotation).count() == 0
        assert db.session.query(MessageChain).count() == 0
        assert db.session.query(MessageAgentThought).count() == 0
        assert db.session.query(MessageFile).count() == 0
        assert db.session.query(SavedMessage).count() == 0
        assert db.session.query(AppAnnotationHitHistory).count() == 0
        assert db.session.query(DatasetRetrieverResource).count() == 0

    def test_clean_mixed_tenants(
        self, db_session_with_containers, mock_billing_service, expired_date, non_expired_date
    ):
        """Test cleaning with mix of sandbox and non-sandbox tenants."""
        # Setup: Create mixed tenants
        sandbox_tenant_ids = []
        pro_tenant_ids = []

        # 2 sandbox tenants with expired messages
        for i in range(2):
            account, tenant = self._create_account_and_tenant(plan="sandbox")
            sandbox_tenant_ids.append(tenant.id)
            app = self._create_app(tenant, account)
            conv = self._create_conversation(app, account)
            self._create_message(app, conv, created_at=expired_date)

        # 2 professional tenants with expired messages (should not be deleted)
        for i in range(2):
            account, tenant = self._create_account_and_tenant(plan="professional")
            pro_tenant_ids.append(tenant.id)
            app = self._create_app(tenant, account)
            conv = self._create_conversation(app, account)
            self._create_message(app, conv, created_at=expired_date)

        # Mock billing service
        plan_map = {}
        for tid in sandbox_tenant_ids:
            plan_map[tid] = CloudPlan.SANDBOX
        for tid in pro_tenant_ids:
            plan_map[tid] = CloudPlan.PROFESSIONAL
        mock_billing_service.get_plan_bulk.return_value = plan_map

        # Execute
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert
        assert stats["batches"] == 1
        assert stats["tenants_processed"] == 2  # Only sandbox tenants
        assert stats["total_messages"] == 2
        assert stats["total_deleted"] == 2

        # Only professional tenant messages should remain
        assert db.session.query(Message).count() == 2
        remaining_messages = db.session.query(Message).all()
        remaining_app_ids = {msg.app_id for msg in remaining_messages}
        pro_app_ids = {app.id for app in db.session.query(App).where(App.tenant_id.in_(pro_tenant_ids)).all()}
        assert remaining_app_ids == pro_app_ids

    def test_clean_mixed_expired_non_expired_messages(
        self, db_session_with_containers, mock_billing_service, expired_date, non_expired_date
    ):
        """Test cleaning sandbox tenant with both expired and non-expired messages."""
        # Setup: Create sandbox tenant with mixed messages
        account, tenant = self._create_account_and_tenant(plan="sandbox")
        app = self._create_app(tenant, account)
        conv = self._create_conversation(app, account)

        # Create 3 expired messages
        for i in range(3):
            self._create_message(app, conv, created_at=expired_date - datetime.timedelta(days=i))

        # Create 2 non-expired messages
        for i in range(2):
            self._create_message(app, conv, created_at=non_expired_date)

        # Mock billing service
        mock_billing_service.get_plan_bulk.return_value = {tenant.id: CloudPlan.SANDBOX}

        # Execute
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert
        assert stats["batches"] == 1
        assert stats["tenants_processed"] == 1
        assert stats["total_messages"] == 3  # Only expired messages
        assert stats["total_deleted"] == 3

        # Non-expired messages should remain
        assert db.session.query(Message).count() == 2
        remaining_messages = db.session.query(Message).all()
        for msg in remaining_messages:
            assert msg.created_at >= expired_date

    def test_clean_large_batch(self, db_session_with_containers, mock_billing_service, expired_date):
        """Test cleaning with large batch of messages."""
        # Setup: Create sandbox tenant with many messages
        account, tenant = self._create_account_and_tenant(plan="sandbox")
        app = self._create_app(tenant, account)
        conv = self._create_conversation(app, account)

        # Create 50 expired messages (will be deleted in batches)
        num_messages = 50
        for i in range(num_messages):
            self._create_message(app, conv, created_at=expired_date - datetime.timedelta(hours=i))

        # Mock billing service
        mock_billing_service.get_plan_bulk.return_value = {tenant.id: CloudPlan.SANDBOX}

        # Execute
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert
        assert stats["batches"] == 1
        assert stats["tenants_processed"] == 1
        assert stats["total_messages"] == num_messages
        assert stats["total_deleted"] == num_messages

        # All messages should be deleted
        assert db.session.query(Message).count() == 0

    def test_clean_no_apps_for_sandbox_tenant(self, db_session_with_containers, mock_billing_service):
        """Test cleaning sandbox tenant with no apps."""
        # Setup: Create sandbox tenant without apps
        account, tenant = self._create_account_and_tenant(plan="sandbox")

        # Mock billing service
        mock_billing_service.get_plan_bulk.return_value = {tenant.id: CloudPlan.SANDBOX}

        # Execute
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert
        assert stats["batches"] == 1
        assert stats["tenants_processed"] == 0  # Skipped due to no apps
        assert stats["total_messages"] == 0
        assert stats["total_deleted"] == 0

    def test_clean_billing_service_error(self, db_session_with_containers, mock_billing_service, expired_date):
        """Test handling of billing service errors."""
        # Setup: Create tenant with expired messages
        account, tenant = self._create_account_and_tenant(plan="sandbox")
        app = self._create_app(tenant, account)
        conv = self._create_conversation(app, account)
        self._create_message(app, conv, created_at=expired_date)

        # Mock billing service to raise exception
        mock_billing_service.get_plan_bulk.side_effect = Exception("Billing API unavailable")

        # Execute - should not raise exception
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert - no tenants should be processed due to billing error
        assert stats["batches"] == 1
        assert stats["tenants_processed"] == 0
        assert stats["total_messages"] == 0
        assert stats["total_deleted"] == 0

        # Message should not be deleted
        assert db.session.query(Message).count() == 1

    def test_clean_billing_service_partial_response(
        self, db_session_with_containers, mock_billing_service, expired_date
    ):
        """Test handling when billing service returns partial data."""
        # Setup: Create 3 tenants
        tenant_ids = []
        for i in range(3):
            account, tenant = self._create_account_and_tenant(plan="sandbox")
            tenant_ids.append(tenant.id)
            app = self._create_app(tenant, account)
            conv = self._create_conversation(app, account)
            self._create_message(app, conv, created_at=expired_date)

        # Mock billing service to return data for only 2 tenants
        mock_billing_service.get_plan_bulk.return_value = {
            tenant_ids[0]: CloudPlan.SANDBOX,
            tenant_ids[1]: CloudPlan.PROFESSIONAL,
            # tenant_ids[2] is missing
        }

        # Execute
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert - only tenant with SANDBOX plan should be processed
        assert stats["batches"] == 1
        assert stats["tenants_processed"] == 1
        assert stats["total_messages"] == 1
        assert stats["total_deleted"] == 1

        # 2 messages should remain (professional and missing tenant)
        assert db.session.query(Message).count() == 2

    def test_clean_multiple_apps_per_tenant(self, db_session_with_containers, mock_billing_service, expired_date):
        """Test cleaning tenant with multiple apps."""
        # Setup: Create sandbox tenant with 3 apps
        account, tenant = self._create_account_and_tenant(plan="sandbox")

        for i in range(3):
            app = self._create_app(tenant, account)
            conv = self._create_conversation(app, account)
            # Create 2 messages per app
            for j in range(2):
                self._create_message(app, conv, created_at=expired_date)

        # Mock billing service
        mock_billing_service.get_plan_bulk.return_value = {tenant.id: CloudPlan.SANDBOX}

        # Execute
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert
        assert stats["batches"] == 1
        assert stats["tenants_processed"] == 1
        assert stats["total_messages"] == 6  # 3 apps * 2 messages
        assert stats["total_deleted"] == 6

        # All messages should be deleted
        assert db.session.query(Message).count() == 0

    def test_clean_preserves_conversation(self, db_session_with_containers, mock_billing_service, expired_date):
        """Test that conversation is preserved even when messages are deleted."""
        # Setup
        account, tenant = self._create_account_and_tenant(plan="sandbox")
        app = self._create_app(tenant, account)
        conv = self._create_conversation(app, account)
        self._create_message(app, conv, created_at=expired_date)

        # Mock billing service
        mock_billing_service.get_plan_bulk.return_value = {tenant.id: CloudPlan.SANDBOX}

        # Execute
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert
        assert stats["total_deleted"] == 1
        assert db.session.query(Message).count() == 0

        # Conversation should still exist
        assert db.session.query(Conversation).count() == 1
        remaining_conv = db.session.query(Conversation).first()
        assert remaining_conv.id == conv.id

    def test_clean_with_cursor_pagination(self, db_session_with_containers, mock_billing_service, expired_date):
        """Test cursor-based pagination works correctly with multiple batches.

        This test validates:
        1. Tenant cursor pagination - processing tenants in multiple batches
        2. Message batch deletion - deleting messages in multiple batches per tenant
        """
        # Mock batch sizes for testing
        dify_config.SANDBOX_MESSAGES_CLEAN_TENANT_BATCH_SIZE = 2
        dify_config.SANDBOX_MESSAGES_CLEAN_MESSAGE_BATCH_SIZE = 2

        # Create enough tenants to trigger multiple tenant batches
        tenant_batch_size = dify_config.SANDBOX_MESSAGES_CLEAN_TENANT_BATCH_SIZE
        num_tenants = int(tenant_batch_size * 2.5)  # 5 tenants = 3 tenant batches (2+2+1)

        # Create enough messages per tenant to trigger multiple message batches
        messages_per_tenant = 5  # 5 messages per tenant = 3 message batches (2+2+1)

        tenant_ids = []
        total_messages_created = 0

        for i in range(num_tenants):
            account, tenant = self._create_account_and_tenant(plan="sandbox")
            tenant_ids.append(tenant.id)
            app = self._create_app(tenant, account)
            conv = self._create_conversation(app, account)

            # Create multiple messages per tenant to test message batch deletion
            for j in range(messages_per_tenant):
                self._create_message(app, conv, created_at=expired_date - datetime.timedelta(hours=j))
                total_messages_created += 1

        # Mock billing service
        mock_billing_service.get_plan_bulk.return_value = dict.fromkeys(tenant_ids, CloudPlan.SANDBOX)

        # Execute
        stats = SandboxMessagesCleanService.clean_sandbox_messages()

        # Assert tenant pagination
        assert stats["batches"] >= 3, (
            f"Expected at least 3 tenant batches (for {num_tenants} tenants with batch size {tenant_batch_size}), "
            f"but got {stats['batches']}"
        )
        assert stats["tenants_processed"] == num_tenants, (
            f"Expected {num_tenants} tenants processed, but got {stats['tenants_processed']}"
        )

        # Assert message batch deletion
        expected_total_messages = num_tenants * messages_per_tenant
        assert stats["total_messages"] == expected_total_messages, (
            f"Expected {expected_total_messages} messages found "
            f"({num_tenants} tenants * {messages_per_tenant} messages), "
            f"but got {stats['total_messages']}"
        )
        assert stats["total_deleted"] == expected_total_messages, (
            f"Expected {expected_total_messages} messages deleted, but got {stats['total_deleted']}"
        )

        # Verify all messages were deleted from database
        remaining_messages = db.session.query(Message).count()
        assert remaining_messages == 0, f"Expected all messages to be deleted, but found {remaining_messages} remaining"

        # Verify all related records were deleted
        assert db.session.query(MessageFeedback).count() == 0
        assert db.session.query(MessageAnnotation).count() == 0
        assert db.session.query(MessageChain).count() == 0
        assert db.session.query(MessageAgentThought).count() == 0
        assert db.session.query(MessageFile).count() == 0
        assert db.session.query(SavedMessage).count() == 0
        assert db.session.query(AppAnnotationHitHistory).count() == 0
        assert db.session.query(DatasetRetrieverResource).count() == 0
