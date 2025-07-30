"""
Performance monitoring utilities for the semiconductor fault detection project.
Tracks execution times, memory usage, and performance metrics.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
import os

class PerformanceMonitor:
    """Monitor performance metrics for the application."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.metrics: Dict[str, Dict[str, Any]] = {}
    
    def start_timer(self, operation: str) -> float:
        """Start timing an operation."""
        start_time = time.time()
        if self.logger:
            self.logger.info(f"⏱️  Starting {operation}")
        return start_time
    
    def end_timer(self, operation: str, start_time: float) -> float:
        """End timing an operation and log the duration."""
        duration = time.time() - start_time
        if self.logger:
            self.logger.info(f"✅ {operation} completed in {duration:.3f}s")
        
        # Store metric
        if operation not in self.metrics:
            self.metrics[operation] = {}
        self.metrics[operation]['duration'] = duration
        self.metrics[operation]['timestamp'] = time.time()
        
        return duration
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    
    def log_memory_usage(self, operation: str = "Current") -> None:
        """Log current memory usage."""
        memory = self.get_memory_usage()
        if self.logger:
            self.logger.info(f"💾 {operation} Memory: {memory['rss_mb']:.1f}MB RSS, {memory['percent']:.1f}%")
    
    @contextmanager
    def monitor_operation(self, operation: str):
        """Context manager to monitor an operation."""
        start_time = self.start_timer(operation)
        initial_memory = self.get_memory_usage()
        
        try:
            yield
        finally:
            duration = self.end_timer(operation, start_time)
            final_memory = self.get_memory_usage()
            
            memory_diff = {
                'rss_diff_mb': final_memory['rss_mb'] - initial_memory['rss_mb'],
                'vms_diff_mb': final_memory['vms_mb'] - initial_memory['vms_mb']
            }
            
            if self.logger:
                self.logger.info(f"📊 {operation} Memory delta: {memory_diff['rss_diff_mb']:+.1f}MB RSS")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of all performance metrics."""
        summary = {
            'total_operations': len(self.metrics),
            'total_duration': sum(metric['duration'] for metric in self.metrics.values()),
            'average_duration': sum(metric['duration'] for metric in self.metrics.values()) / len(self.metrics) if self.metrics else 0,
            'operations': self.metrics,
            'current_memory': self.get_memory_usage()
        }
        return summary
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        self.metrics.clear()

def monitor_performance(operation_name: Optional[str] = None):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Try to get logger from first argument if it's a method
            logger = None
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            
            monitor = PerformanceMonitor(logger)
            
            with monitor.monitor_operation(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global performance monitor instance
performance_monitor = PerformanceMonitor() 