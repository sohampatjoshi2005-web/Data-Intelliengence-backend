"""
Async Task Queue System for Long-Running ML Operations
Prevents blocking API responses during training, tuning, and processing
Uses in-memory task storage for development; can be upgraded to Redis/Celery
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Callable, Optional
from enum import Enum
from dataclasses import dataclass, asdict
import traceback

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0-100
    created_at: str = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
    
    def to_dict(self):
        return asdict(self)


# In-memory task storage (upgrade to Redis for production)
_tasks: Dict[str, Task] = {}
_tasks_lock = asyncio.Lock()


async def create_task(func: Callable, *args, **kwargs) -> str:
    """
    Queue a long-running function and return task ID immediately
    Function runs in background, client polls /tasks/{task_id} for progress
    
    Usage:
        task_id = await create_task(run_optuna_tuning, df, target="y", task="regression")
        return {"task_id": task_id, "status": "queued"}
    """
    task_id = str(uuid.uuid4())
    task = Task(task_id=task_id)
    
    async with _tasks_lock:
        _tasks[task_id] = task
    
    # Run task in background (non-blocking)
    asyncio.create_task(_execute_task(task_id, func, args, kwargs))
    
    return task_id


async def _execute_task(task_id: str, func: Callable, args: tuple, kwargs: dict):
    """Background task execution"""
    task = _tasks[task_id]
    
    try:
        # Update task status
        async with _tasks_lock:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow().isoformat()
        
        # Run function (assumes it's fast enough or has internal progress)
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            # Run sync function in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, func, *args, **kwargs)
        
        # Store result
        async with _tasks_lock:
            task.result = result
            task.status = TaskStatus.SUCCESS
            task.progress = 100.0
            task.completed_at = datetime.utcnow().isoformat()
            
    except Exception as e:
        # Store error
        async with _tasks_lock:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow().isoformat()
        print(f"Task {task_id} failed: {traceback.format_exc()}")


async def get_task(task_id: str) -> Optional[Task]:
    """Get task status and result"""
    async with _tasks_lock:
        return _tasks.get(task_id)


async def update_task_progress(task_id: str, progress: float):
    """Update task progress (0-100)"""
    async with _tasks_lock:
        if task_id in _tasks:
            _tasks[task_id].progress = min(100.0, max(0.0, progress))


def get_task_sync(task_id: str) -> Optional[Task]:
    """Synchronous get task for use in endpoints"""
    return _tasks.get(task_id)


async def cancel_task(task_id: str) -> bool:
    """Cancel a pending task"""
    async with _tasks_lock:
        if task_id in _tasks:
            task = _tasks[task_id]
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                return True
    return False
