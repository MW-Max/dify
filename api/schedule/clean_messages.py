import logging
import time

import click

import app
from services.sandbox_messages_clean_service import SandboxMessagesCleanService

logger = logging.getLogger(__name__)


@app.celery.task(queue="retention")
def clean_messages():
    """
    Clean expired messages from sandbox plan tenants.
    
    This task uses SandboxMessagesCleanService to efficiently clean messages in batches.
    """
    click.echo(click.style("clean_messages: start clean messages.", fg="green"))
    start_at = time.perf_counter()
    
    try:
        stats = SandboxMessagesCleanService.clean_sandbox_messages()
        
        end_at = time.perf_counter()
        click.echo(
            click.style(
                f"clean_messages: completed successfully\n"
                f"  - Latency: {end_at - start_at:.2f}s\n"
                f"  - Batches processed: {stats['batches']}\n"
                f"  - Sandbox tenants: {stats['tenants_processed']}\n"
                f"  - Messages found: {stats['total_messages']}\n"
                f"  - Messages deleted: {stats['total_deleted']}",
                fg="green",
            )
        )
    except Exception as e:
        end_at = time.perf_counter()
        logger.exception("clean_messages failed")
        click.echo(
            click.style(
                f"clean_messages: failed after {end_at - start_at:.2f}s - {str(e)}",
                fg="red",
            )
        )
        raise
