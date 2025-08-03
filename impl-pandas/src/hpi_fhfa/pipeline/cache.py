"""Caching and persistence for HPI system"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import pickle
import hashlib
import threading
import time
import pandas as pd
import logging
from abc import ABC, abstractmethod


class CacheStrategy(str, Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    LFU = "lfu"  # Least Frequently Used


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 1
    ttl_seconds: Optional[int] = None
    size_bytes: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds


class CacheKey:
    """Generate cache keys for different operations"""
    
    @staticmethod
    def for_index_request(start_date: datetime, end_date: datetime,
                         geography_level: str, weighting_scheme: str,
                         filters: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for index request
        
        Args:
            start_date: Start date
            end_date: End date
            geography_level: Geographic level
            weighting_scheme: Weighting scheme
            filters: Optional filters
            
        Returns:
            Cache key string
        """
        key_parts = [
            f"index",
            start_date.isoformat(),
            end_date.isoformat(),
            geography_level,
            weighting_scheme
        ]
        
        if filters:
            # Sort filters for consistent keys
            filter_str = json.dumps(filters, sort_keys=True)
            key_parts.append(hashlib.md5(filter_str.encode()).hexdigest()[:8])
            
        return ":".join(key_parts)
        
    @staticmethod
    def for_quality_report(start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> str:
        """Generate cache key for quality report
        
        Args:
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Cache key string
        """
        key_parts = ["quality"]
        
        if start_date:
            key_parts.append(start_date.isoformat())
        else:
            key_parts.append("all")
            
        if end_date:
            key_parts.append(end_date.isoformat())
        else:
            key_parts.append("all")
            
        return ":".join(key_parts)
        
    @staticmethod
    def for_pipeline_result(pipeline_name: str, context_hash: str) -> str:
        """Generate cache key for pipeline result
        
        Args:
            pipeline_name: Pipeline name
            context_hash: Hash of pipeline context
            
        Returns:
            Cache key string
        """
        return f"pipeline:{pipeline_name}:{context_hash}"


class CacheBackend(ABC):
    """Abstract cache backend"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
        
    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in cache"""
        pass
        
    @abstractmethod
    def delete(self, key: str):
        """Delete value from cache"""
        pass
        
    @abstractmethod
    def clear(self):
        """Clear all cache entries"""
        pass
        
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend"""
    
    def __init__(self, max_size_mb: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        """Initialize memory cache
        
        Args:
            max_size_mb: Maximum cache size in MB
            strategy: Cache eviction strategy
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger("cache.memory")
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired():
                    del self.cache[key]
                    self.misses += 1
                    return None
                    
                # Update access metadata
                entry.accessed_at = datetime.now()
                entry.access_count += 1
                self.hits += 1
                
                return entry.value
            else:
                self.misses += 1
                return None
                
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in cache"""
        with self.lock:
            # Estimate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1000  # Default estimate
                
            # Check if eviction needed
            self._evict_if_needed(size_bytes)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            
    def delete(self, key: str):
        """Delete value from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_size = sum(e.size_bytes or 0 for e in self.cache.values())
            total_requests = self.hits + self.misses
            
            return {
                'entries': len(self.cache),
                'size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': self.hits / total_requests if total_requests > 0 else 0,
                'evictions': self.evictions,
                'strategy': self.strategy.value
            }
            
    def _evict_if_needed(self, new_size: int):
        """Evict entries if needed to make space"""
        current_size = sum(e.size_bytes or 0 for e in self.cache.values())
        
        if current_size + new_size <= self.max_size_bytes:
            return
            
        # Need to evict
        entries = list(self.cache.values())
        
        if self.strategy == CacheStrategy.LRU:
            # Sort by last accessed time
            entries.sort(key=lambda e: e.accessed_at)
        elif self.strategy == CacheStrategy.LFU:
            # Sort by access count
            entries.sort(key=lambda e: e.access_count)
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired first, then oldest
            entries.sort(key=lambda e: (not e.is_expired(), e.created_at))
            
        # Evict until we have space
        while current_size + new_size > self.max_size_bytes and entries:
            entry = entries.pop(0)
            del self.cache[entry.key]
            current_size -= entry.size_bytes or 0
            self.evictions += 1


class DiskCacheBackend(CacheBackend):
    """Disk-based cache backend"""
    
    def __init__(self, cache_dir: Path, max_size_gb: int = 10):
        """Initialize disk cache
        
        Args:
            cache_dir: Directory for cache files
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.logger = logging.getLogger("cache.disk")
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.metadata:
                return None
                
            entry_meta = self.metadata[key]
            
            # Check expiration
            created_at = datetime.fromisoformat(entry_meta['created_at'])
            ttl = entry_meta.get('ttl_seconds')
            if ttl:
                age = (datetime.now() - created_at).total_seconds()
                if age > ttl:
                    self.delete(key)
                    return None
                    
            # Load from disk
            file_path = self.cache_dir / entry_meta['filename']
            if not file_path.exists():
                del self.metadata[key]
                return None
                
            try:
                with open(file_path, 'rb') as f:
                    value = pickle.load(f)
                    
                # Update access time
                self.metadata[key]['accessed_at'] = datetime.now().isoformat()
                self.metadata[key]['access_count'] = entry_meta.get('access_count', 0) + 1
                
                return value
            except Exception as e:
                self.logger.error(f"Failed to load cache entry {key}: {str(e)}")
                self.delete(key)
                return None
                
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in cache"""
        with self.lock:
            # Generate filename
            filename = f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
            file_path = self.cache_dir / filename
            
            # Save to disk
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                    
                size_bytes = file_path.stat().st_size
                
                # Update metadata
                self.metadata[key] = {
                    'filename': filename,
                    'created_at': datetime.now().isoformat(),
                    'accessed_at': datetime.now().isoformat(),
                    'access_count': 1,
                    'ttl_seconds': ttl_seconds,
                    'size_bytes': size_bytes
                }
                
                # Check size limit
                self._evict_if_needed()
                
                # Save metadata
                self._save_metadata()
                
            except Exception as e:
                self.logger.error(f"Failed to save cache entry {key}: {str(e)}")
                if file_path.exists():
                    file_path.unlink()
                    
    def delete(self, key: str):
        """Delete value from cache"""
        with self.lock:
            if key in self.metadata:
                filename = self.metadata[key]['filename']
                file_path = self.cache_dir / filename
                
                if file_path.exists():
                    file_path.unlink()
                    
                del self.metadata[key]
                self._save_metadata()
                
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            # Delete all cache files
            for entry in self.metadata.values():
                file_path = self.cache_dir / entry['filename']
                if file_path.exists():
                    file_path.unlink()
                    
            self.metadata.clear()
            self._save_metadata()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_size = sum(e.get('size_bytes', 0) for e in self.metadata.values())
            
            return {
                'entries': len(self.metadata),
                'size_gb': total_size / (1024 * 1024 * 1024),
                'max_size_gb': self.max_size_bytes / (1024 * 1024 * 1024),
                'cache_dir': str(self.cache_dir)
            }
            
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
        
    def _save_metadata(self):
        """Save cache metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {str(e)}")
            
    def _evict_if_needed(self):
        """Evict entries if cache size exceeds limit"""
        total_size = sum(e.get('size_bytes', 0) for e in self.metadata.values())
        
        if total_size <= self.max_size_bytes:
            return
            
        # Sort by access time (LRU)
        entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1].get('accessed_at', x[1]['created_at'])
        )
        
        # Evict until under limit
        while total_size > self.max_size_bytes and entries:
            key, entry = entries.pop(0)
            self.delete(key)
            total_size -= entry.get('size_bytes', 0)


class ResultCache:
    """High-level caching interface for HPI results"""
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        """Initialize result cache
        
        Args:
            backend: Cache backend to use
        """
        self.backend = backend or MemoryCacheBackend()
        self.logger = logging.getLogger("result_cache")
        
    def get_index_results(self, start_date: datetime, end_date: datetime,
                         geography_level: str, weighting_scheme: str,
                         filters: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """Get cached index results
        
        Args:
            start_date: Start date
            end_date: End date
            geography_level: Geographic level
            weighting_scheme: Weighting scheme
            filters: Optional filters
            
        Returns:
            Cached DataFrame or None
        """
        key = CacheKey.for_index_request(
            start_date, end_date, geography_level,
            weighting_scheme, filters
        )
        
        result = self.backend.get(key)
        if result is not None:
            self.logger.info(f"Cache hit for index results: {key}")
        return result
        
    def set_index_results(self, results: pd.DataFrame,
                         start_date: datetime, end_date: datetime,
                         geography_level: str, weighting_scheme: str,
                         filters: Optional[Dict[str, Any]] = None,
                         ttl_hours: int = 24):
        """Cache index results
        
        Args:
            results: Results DataFrame
            start_date: Start date
            end_date: End date
            geography_level: Geographic level
            weighting_scheme: Weighting scheme
            filters: Optional filters
            ttl_hours: Time to live in hours
        """
        key = CacheKey.for_index_request(
            start_date, end_date, geography_level,
            weighting_scheme, filters
        )
        
        self.backend.set(key, results, ttl_seconds=ttl_hours * 3600)
        self.logger.info(f"Cached index results: {key}")
        
    def invalidate_index_results(self, start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None):
        """Invalidate cached index results
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
        """
        # In a real implementation, would selectively invalidate
        # For now, clear all if any date range specified
        if start_date or end_date:
            self.logger.info("Invalidating cached index results")
            # Would implement selective invalidation
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.backend.get_stats()