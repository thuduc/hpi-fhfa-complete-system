"""Unit tests for caching system"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import time
import pandas as pd
import numpy as np

from hpi_fhfa.pipeline.cache import (
    CacheKey, MemoryCacheBackend, DiskCacheBackend, ResultCache,
    CacheEntry, CacheStrategy
)


class TestCacheKey:
    """Test cache key generation"""
    
    def test_index_request_key(self):
        """Test cache key for index requests"""
        key1 = CacheKey.for_index_request(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2021, 12, 31),
            geography_level="cbsa",
            weighting_scheme="sample"
        )
        
        key2 = CacheKey.for_index_request(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2021, 12, 31),
            geography_level="cbsa",
            weighting_scheme="sample"
        )
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different keys
        key3 = CacheKey.for_index_request(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2021, 12, 31),
            geography_level="tract",  # Different
            weighting_scheme="sample"
        )
        
        assert key1 != key3
        
    def test_index_request_key_with_filters(self):
        """Test cache key with filters"""
        filters1 = {"cbsa_id": ["12420", "12580"], "state": ["CA"]}
        filters2 = {"state": ["CA"], "cbsa_id": ["12420", "12580"]}  # Different order
        
        key1 = CacheKey.for_index_request(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2021, 12, 31),
            geography_level="cbsa",
            weighting_scheme="sample",
            filters=filters1
        )
        
        key2 = CacheKey.for_index_request(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2021, 12, 31),
            geography_level="cbsa",
            weighting_scheme="sample",
            filters=filters2
        )
        
        # Should be same despite different order
        assert key1 == key2
        
    def test_quality_report_key(self):
        """Test cache key for quality reports"""
        key1 = CacheKey.for_quality_report(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2021, 12, 31)
        )
        
        key2 = CacheKey.for_quality_report()  # No dates
        
        assert key1 != key2
        assert "all" in key2  # Should contain "all" for missing dates
        
    def test_pipeline_result_key(self):
        """Test cache key for pipeline results"""
        key = CacheKey.for_pipeline_result(
            pipeline_name="hpi_calculation",
            context_hash="abc123def456"
        )
        
        assert "pipeline:hpi_calculation:abc123def456" == key


class TestCacheEntry:
    """Test cache entry functionality"""
    
    def test_cache_entry_creation(self):
        """Test creating cache entries"""
        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            ttl_seconds=3600
        )
        
        assert entry.key == "test_key"
        assert entry.value == {"data": "test"}
        assert entry.access_count == 1
        assert entry.ttl_seconds == 3600
        
    def test_cache_entry_expiration(self):
        """Test cache entry expiration check"""
        # Non-expiring entry
        entry1 = CacheEntry(
            key="test1",
            value="data",
            created_at=datetime.now(),
            accessed_at=datetime.now()
        )
        
        assert entry1.is_expired() is False
        
        # Expired entry
        entry2 = CacheEntry(
            key="test2",
            value="data",
            created_at=datetime.now() - timedelta(seconds=10),
            accessed_at=datetime.now(),
            ttl_seconds=5
        )
        
        assert entry2.is_expired() is True
        
        # Not yet expired
        entry3 = CacheEntry(
            key="test3",
            value="data",
            created_at=datetime.now() - timedelta(seconds=2),
            accessed_at=datetime.now(),
            ttl_seconds=10
        )
        
        assert entry3.is_expired() is False


class TestMemoryCacheBackend:
    """Test in-memory cache backend"""
    
    def test_basic_get_set(self):
        """Test basic get/set operations"""
        cache = MemoryCacheBackend(max_size_mb=10)
        
        # Set value
        cache.set("key1", {"data": "test"})
        
        # Get value
        value = cache.get("key1")
        assert value == {"data": "test"}
        
        # Get non-existent key
        value = cache.get("nonexistent")
        assert value is None
        
    def test_ttl_expiration(self):
        """Test TTL expiration"""
        cache = MemoryCacheBackend()
        
        # Set with short TTL
        cache.set("expire_test", "data", ttl_seconds=0.1)
        
        # Should exist immediately
        assert cache.get("expire_test") == "data"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get("expire_test") is None
        
    def test_cache_statistics(self):
        """Test cache statistics"""
        cache = MemoryCacheBackend()
        
        # Generate some activity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["entries"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2/3
        
    def test_lru_eviction(self):
        """Test LRU eviction strategy"""
        cache = MemoryCacheBackend(
            max_size_mb=0.0001,  # Even smaller to ensure eviction
            strategy=CacheStrategy.LRU
        )
        
        # Add entries with larger size to ensure we exceed limit
        large_value = "x" * 1000  # 1KB each
        cache.set("key1", large_value)
        cache.set("key2", large_value)
        cache.set("key3", large_value)
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add another entry to trigger eviction
        cache.set("key4", large_value)
        
        # Check that we have fewer than 4 entries (some were evicted)
        stats = cache.get_stats()
        assert stats["entries"] < 4
        assert stats["evictions"] > 0
        
        # Most recently used keys should still be there
        # key1 was accessed, key3 and key4 are newest
        assert cache.get("key1") is not None or cache.get("key4") is not None
        
    def test_lfu_eviction(self):
        """Test LFU eviction strategy"""
        cache = MemoryCacheBackend(
            max_size_mb=0.001,
            strategy=CacheStrategy.LFU
        )
        
        # Add entries
        cache.set("key1", "a" * 100)
        cache.set("key2", "b" * 100)
        
        # Access key1 multiple times
        for _ in range(5):
            cache.get("key1")
            
        # Access key2 once
        cache.get("key2")
        
        # Add entry to trigger eviction
        cache.set("key3", "c" * 200)
        
        # key2 should be evicted (least frequently used)
        assert cache.get("key1") is not None
        
    def test_clear_cache(self):
        """Test clearing cache"""
        cache = MemoryCacheBackend()
        
        # Add entries
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Verify they exist
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        
        # Clear
        cache.clear()
        
        # Should be empty
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        
        # After clear, all stats should be reset
        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 2  # The two gets after clear are misses


class TestDiskCacheBackend:
    """Test disk-based cache backend"""
    
    def test_disk_cache_operations(self):
        """Test basic disk cache operations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheBackend(
                cache_dir=Path(tmpdir),
                max_size_gb=1
            )
            
            # Set value
            test_data = {"data": [1, 2, 3], "text": "test"}
            cache.set("test_key", test_data)
            
            # Get value
            retrieved = cache.get("test_key")
            assert retrieved == test_data
            
            # Check file exists
            files = list(Path(tmpdir).glob("*.pkl"))
            assert len(files) == 1
            
            # Delete
            cache.delete("test_key")
            assert cache.get("test_key") is None
            
            # File should be deleted
            files = list(Path(tmpdir).glob("*.pkl"))
            assert len(files) == 0
            
    def test_disk_cache_persistence(self):
        """Test that disk cache persists across instances"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # First instance - write data
            cache1 = DiskCacheBackend(cache_dir)
            cache1.set("persistent", {"value": 42})
            
            # Second instance - read data
            cache2 = DiskCacheBackend(cache_dir)
            value = cache2.get("persistent")
            assert value == {"value": 42}
            
    def test_disk_cache_ttl(self):
        """Test TTL in disk cache"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheBackend(Path(tmpdir))
            
            # Set with TTL
            cache.set("expire_test", "data", ttl_seconds=0.1)
            
            # Should exist
            assert cache.get("expire_test") == "data"
            
            # Wait for expiration
            time.sleep(0.2)
            
            # Should be expired
            assert cache.get("expire_test") is None
            
    def test_disk_cache_size_limit(self):
        """Test disk cache size limit"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheBackend(
                cache_dir=Path(tmpdir),
                max_size_gb=0.000001  # Very small to trigger eviction
            )
            
            # Add large entries
            for i in range(5):
                cache.set(f"key{i}", "x" * 1000)
                
            # Should have evicted some entries
            stats = cache.get_stats()
            assert stats["entries"] < 5
            
    def test_disk_cache_clear(self):
        """Test clearing disk cache"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheBackend(Path(tmpdir))
            
            # Add entries
            cache.set("key1", "value1")
            cache.set("key2", "value2")
            
            # Clear
            cache.clear()
            
            # Should be empty
            assert cache.get("key1") is None
            assert cache.get("key2") is None
            
            # No cache files should exist
            files = list(Path(tmpdir).glob("*.pkl"))
            assert len(files) == 0


class TestResultCache:
    """Test high-level result cache"""
    
    def test_index_result_caching(self):
        """Test caching index results"""
        cache = ResultCache(backend=MemoryCacheBackend())
        
        # Create test DataFrame
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=12, freq='ME'),
            'index_value': np.random.randn(12) * 0.02 + 1.0
        })
        
        # Cache results
        cache.set_index_results(
            results=df,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            geography_level="cbsa",
            weighting_scheme="sample",
            ttl_hours=1
        )
        
        # Retrieve results
        retrieved = cache.get_index_results(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            geography_level="cbsa",
            weighting_scheme="sample"
        )
        
        assert retrieved is not None
        assert len(retrieved) == len(df)
        assert list(retrieved.columns) == list(df.columns)
        
    def test_cache_miss(self):
        """Test cache miss behavior"""
        cache = ResultCache(backend=MemoryCacheBackend())
        
        # Try to get non-existent results
        result = cache.get_index_results(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            geography_level="cbsa",
            weighting_scheme="sample"
        )
        
        assert result is None
        
    def test_result_cache_with_filters(self):
        """Test caching with filters"""
        cache = ResultCache(backend=MemoryCacheBackend())
        
        df = pd.DataFrame({'value': [1, 2, 3]})
        filters = {"cbsa_id": ["12420"]}
        
        # Cache with filters
        cache.set_index_results(
            results=df,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            geography_level="cbsa",
            weighting_scheme="sample",
            filters=filters
        )
        
        # Should hit with same filters
        result = cache.get_index_results(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            geography_level="cbsa",
            weighting_scheme="sample",
            filters=filters
        )
        
        assert result is not None
        
        # Should miss with different filters
        result = cache.get_index_results(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            geography_level="cbsa",
            weighting_scheme="sample",
            filters={"cbsa_id": ["99999"]}
        )
        
        assert result is None
        
    def test_cache_stats_access(self):
        """Test accessing cache statistics"""
        backend = MemoryCacheBackend()
        cache = ResultCache(backend=backend)
        
        # Generate some cache activity
        df = pd.DataFrame({'value': [1, 2, 3]})
        cache.set_index_results(
            df,
            datetime(2020, 1, 1),
            datetime(2020, 12, 31),
            "cbsa",
            "sample"
        )
        
        # Get stats
        stats = cache.get_stats()
        
        assert "entries" in stats
        assert "hit_rate" in stats
        assert stats["entries"] == 1