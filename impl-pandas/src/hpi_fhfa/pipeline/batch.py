"""Batch processing capabilities for HPI system"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
import uuid

from .orchestrator import Pipeline, PipelineResult


class BatchJobStatus(str, Enum):
    """Batch job status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Job priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BatchJob:
    """Batch job definition"""
    job_id: str
    name: str
    pipeline: str
    context: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: BatchJobStatus = BatchJobStatus.QUEUED
    result: Optional[PipelineResult] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchQueue:
    """Priority queue for batch jobs"""
    
    def __init__(self):
        """Initialize batch queue"""
        self._queue = queue.PriorityQueue()
        self._jobs: Dict[str, BatchJob] = {}
        self._lock = threading.Lock()
        
    def add_job(self, job: BatchJob):
        """Add job to queue
        
        Args:
            job: Batch job to add
        """
        with self._lock:
            # Priority queue uses negative priority for higher values first
            priority_value = -job.priority.value
            self._queue.put((priority_value, job.created_at, job.job_id))
            self._jobs[job.job_id] = job
            
    def get_job(self, timeout: Optional[float] = None) -> Optional[BatchJob]:
        """Get next job from queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Next job or None if timeout
        """
        try:
            _, _, job_id = self._queue.get(timeout=timeout)
            with self._lock:
                return self._jobs.get(job_id)
        except queue.Empty:
            return None
            
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancelled, False if not found or already running
        """
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if job.status == BatchJobStatus.QUEUED:
                    job.status = BatchJobStatus.CANCELLED
                    job.completed_at = datetime.now()
                    return True
        return False
        
    def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get job status
        
        Args:
            job_id: Job ID to check
            
        Returns:
            Job object or None if not found
        """
        with self._lock:
            return self._jobs.get(job_id)
            
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._lock:
            jobs_by_status = {}
            for job in self._jobs.values():
                status = job.status.value
                jobs_by_status[status] = jobs_by_status.get(status, 0) + 1
                
            return {
                'total_jobs': len(self._jobs),
                'queued': jobs_by_status.get(BatchJobStatus.QUEUED.value, 0),
                'running': jobs_by_status.get(BatchJobStatus.RUNNING.value, 0),
                'completed': jobs_by_status.get(BatchJobStatus.COMPLETED.value, 0),
                'failed': jobs_by_status.get(BatchJobStatus.FAILED.value, 0),
                'cancelled': jobs_by_status.get(BatchJobStatus.CANCELLED.value, 0)
            }


class BatchProcessor:
    """Batch processing engine"""
    
    def __init__(self,
                 max_workers: int = 4,
                 job_timeout: int = 3600,
                 result_path: Optional[Path] = None):
        """Initialize batch processor
        
        Args:
            max_workers: Maximum concurrent workers
            job_timeout: Job timeout in seconds
            result_path: Path to store job results
        """
        self.max_workers = max_workers
        self.job_timeout = job_timeout
        self.result_path = result_path or Path("batch_results")
        self.result_path.mkdir(parents=True, exist_ok=True)
        
        self.queue = BatchQueue()
        self.pipelines: Dict[str, Pipeline] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: Dict[str, Future] = {}
        self.logger = logging.getLogger("batch_processor")
        
        self._running = False
        self._monitor_thread = None
        
    def register_pipeline(self, name: str, pipeline: Pipeline):
        """Register a pipeline for batch processing
        
        Args:
            name: Pipeline name
            pipeline: Pipeline instance
        """
        self.pipelines[name] = pipeline
        
    def submit_job(self, job: BatchJob) -> str:
        """Submit job for processing
        
        Args:
            job: Batch job to submit
            
        Returns:
            Job ID
        """
        if not job.job_id:
            job.job_id = str(uuid.uuid4())
            
        self.queue.add_job(job)
        self.logger.info(f"Job {job.job_id} ({job.name}) submitted")
        
        # Start processing if not already running
        if not self._running:
            self.start()
            
        return job.job_id
        
    def submit_batch(self, jobs: List[BatchJob]) -> List[str]:
        """Submit multiple jobs
        
        Args:
            jobs: List of batch jobs
            
        Returns:
            List of job IDs
        """
        job_ids = []
        for job in jobs:
            job_id = self.submit_job(job)
            job_ids.append(job_id)
        return job_ids
        
    def start(self):
        """Start batch processor"""
        if self._running:
            return
            
        self._running = True
        self.logger.info("Starting batch processor")
        
        # Start worker threads directly (not through executor)
        self._worker_threads = []
        for i in range(self.max_workers):
            thread = threading.Thread(target=self._worker, args=(i,))
            thread.start()
            self._worker_threads.append(thread)
            
        # Start monitor thread
        self._monitor_thread = threading.Thread(target=self._monitor_jobs)
        self._monitor_thread.start()
        
    def stop(self, wait: bool = True):
        """Stop batch processor
        
        Args:
            wait: Whether to wait for running jobs to complete
        """
        self.logger.info("Stopping batch processor")
        self._running = False
        
        if wait:
            # Wait for worker threads to finish
            for thread in getattr(self, '_worker_threads', []):
                thread.join()
        
        # Always wait for monitor thread
        if self._monitor_thread:
            self._monitor_thread.join()
            
        # Shutdown executor
        self.executor.shutdown(wait=wait)
            
    def _worker(self, worker_id: int):
        """Worker thread function
        
        Args:
            worker_id: Worker ID for logging
        """
        self.logger.info(f"Worker {worker_id} started")
        
        while self._running:
            # Get next job
            job = self.queue.get_job(timeout=1.0)
            if not job:
                continue
                
            if job.status != BatchJobStatus.QUEUED:
                continue
                
            try:
                # Update job status
                job.status = BatchJobStatus.RUNNING
                job.started_at = datetime.now()
                
                self.logger.info(f"Worker {worker_id} processing job {job.job_id}")
                
                # Get pipeline
                pipeline = self.pipelines.get(job.pipeline)
                if not pipeline:
                    raise ValueError(f"Pipeline '{job.pipeline}' not registered")
                
                # Execute pipeline (timeout is handled by monitor thread)
                result = pipeline.execute(job.context)
                
                # Update job with result
                job.status = BatchJobStatus.COMPLETED
                job.completed_at = datetime.now()
                job.result = result
                
                # Save result
                self._save_job_result(job)
                
                self.logger.info(f"Job {job.job_id} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Job {job.job_id} failed: {str(e)}")
                
                job.error = str(e)
                job.completed_at = datetime.now()
                
                # Check retry
                if job.retries < job.max_retries:
                    job.retries += 1
                    job.status = BatchJobStatus.QUEUED
                    job.started_at = None
                    job.completed_at = None
                    self.queue.add_job(job)  # Re-queue
                    self.logger.info(f"Job {job.job_id} re-queued (retry {job.retries})")
                else:
                    job.status = BatchJobStatus.FAILED
                    self._save_job_result(job)
                    
        self.logger.info(f"Worker {worker_id} stopped")
        
    def _monitor_jobs(self):
        """Monitor running jobs for timeouts and cleanup"""
        while self._running:
            try:
                # Check for timed out jobs
                current_time = datetime.now()
                
                stats = self.queue.get_queue_stats()
                if stats['running'] > 0:
                    # Check each running job
                    for job_id, future in list(self.futures.items()):
                        job = self.queue.get_job_status(job_id)
                        if job and job.status == BatchJobStatus.RUNNING:
                            if job.started_at:
                                runtime = (current_time - job.started_at).total_seconds()
                                if runtime > self.job_timeout:
                                    # Cancel job
                                    future.cancel()
                                    job.status = BatchJobStatus.FAILED
                                    job.error = "Job timed out"
                                    job.completed_at = current_time
                                    self._save_job_result(job)
                                    self.logger.warning(f"Job {job_id} timed out")
                
                # Log stats periodically
                self.logger.info(f"Queue stats: {stats}")
                
            except Exception as e:
                self.logger.error(f"Monitor error: {str(e)}")
                
            time.sleep(10)  # Check every 10 seconds
            
    def _save_job_result(self, job: BatchJob):
        """Save job result to disk
        
        Args:
            job: Completed job
        """
        try:
            result_file = self.result_path / f"{job.job_id}.json"
            
            result_data = {
                'job_id': job.job_id,
                'name': job.name,
                'status': job.status.value,
                'created_at': job.created_at.isoformat(),
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'error': job.error,
                'metadata': job.metadata
            }
            
            if job.result:
                result_data['result'] = {
                    'pipeline_id': job.result.pipeline_id,
                    'status': job.result.status,
                    'metrics': job.result.metrics,
                    'error_summary': job.result.error_summary
                }
                
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save job result: {str(e)}")
            
    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job result
        
        Args:
            job_id: Job ID
            
        Returns:
            Job result data or None
        """
        result_file = self.result_path / f"{job_id}.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                return json.load(f)
        return None
        
    def create_recurring_job(self,
                           name: str,
                           pipeline: str,
                           context_generator: Callable[[], Dict[str, Any]],
                           schedule: timedelta,
                           priority: JobPriority = JobPriority.NORMAL) -> str:
        """Create a recurring job
        
        Args:
            name: Job name
            pipeline: Pipeline to run
            context_generator: Function to generate context
            schedule: Time between runs
            priority: Job priority
            
        Returns:
            Recurring job ID
        """
        recurring_id = str(uuid.uuid4())
        
        def schedule_next():
            if self._running:
                try:
                    context = context_generator()
                    job = BatchJob(
                        job_id=f"{recurring_id}_{datetime.now().timestamp()}",
                        name=f"{name} (recurring)",
                        pipeline=pipeline,
                        context=context,
                        priority=priority,
                        metadata={'recurring_id': recurring_id}
                    )
                    self.submit_job(job)
                except Exception as e:
                    self.logger.error(f"Failed to schedule recurring job: {str(e)}")
                
                # Schedule next run
                threading.Timer(schedule.total_seconds(), schedule_next).start()
        
        # Schedule first run
        schedule_next()
        
        return recurring_id