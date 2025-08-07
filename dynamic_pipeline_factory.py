#!/usr/bin/env python3
"""
Dynamic Pipeline Factory for Text-to-Audio Processing

Allows easy experimentation with different text encoders and configurations.
"""

import torch
from typing import Dict, Any, Optional
from pipeline import TextToAudioProcessingPipeline


class DynamicPipelineFactory:
    """
    팩토리 클래스로 다양한 텍스트 인코더를 쉽게 실험할 수 있도록 지원
    """
    
    # 사전 정의된 인코더 설정
    ENCODER_PRESETS = {
        'sentence-transformer-mini': {
            'type': 'sentence-transformer',
            'config': {'model_name': 'all-MiniLM-L6-v2'},
            'description': '빠르고 가벼운 범용 인코더 (384D)'
        },
        'sentence-transformer-large': {
            'type': 'sentence-transformer', 
            'config': {'model_name': 'all-mpnet-base-v2'},
            'description': '고품질 범용 인코더 (768D)'
        },
        'e5-large': {
            'type': 'e5-large',
            'config': {'model_name': 'intfloat/e5-large-v2'},
            'description': '최고 성능 오픈소스 인코더 (1024D)'
        },
        'clap': {
            'type': 'clap',
            'config': {'model_name': '630k-audioset-best'},
            'description': '오디오-텍스트 특화 인코더 (512D)'
        }
    }
    
    @classmethod
    def create_pipeline(cls, 
                       encoder_preset: str = 'sentence-transformer-large',
                       custom_encoder_config: Optional[Dict] = None,
                       backbone_type: str = 'simple',
                       backbone_config: Optional[Dict] = None,
                       use_clap: bool = True,
                       **pipeline_kwargs) -> TextToAudioProcessingPipeline:
        """
        동적으로 파이프라인 생성 - 백본도 동적으로 지원
        
        Args:
            encoder_preset: 사전 정의된 인코더 설정 이름
            custom_encoder_config: 커스텀 인코더 설정 (preset 오버라이드)
            backbone_type: 백본 타입 ('simple', 'dynamic', 'transformer', etc.)
            backbone_config: 백본 모델 설정
            use_clap: CLAP 인코더 사용 여부
            **pipeline_kwargs: 추가 파이프라인 설정
            
        Returns:
            TextToAudioProcessingPipeline 인스턴스
        """
        
        # 인코더 설정 결정
        if custom_encoder_config:
            encoder_type = custom_encoder_config['type']
            encoder_config = custom_encoder_config.get('config', {})
            print(f"🎯 Using custom encoder: {encoder_type}")
        elif encoder_preset in cls.ENCODER_PRESETS:
            preset = cls.ENCODER_PRESETS[encoder_preset]
            encoder_type = preset['type']
            encoder_config = preset['config']
            print(f"🎯 Using preset '{encoder_preset}': {preset['description']}")
        else:
            raise ValueError(f"Unknown encoder preset: {encoder_preset}. "
                           f"Available: {list(cls.ENCODER_PRESETS.keys())}")
        
        # 백본 설정 준비 - 동적 차원 지원
        if backbone_config is None:
            backbone_config = {}
        
        # 텍스트 인코더 차원 자동 설정
        text_dim_map = {
            'sentence-transformer-mini': 384,
            'sentence-transformer-large': 768,
            'e5-large': 1024,
            'clap': 512
        }
        
        # 백본에 동적 차원 정보 전달
        if encoder_preset in text_dim_map:
            backbone_config['text_dim'] = text_dim_map[encoder_preset]
        
        if use_clap:
            backbone_config['clap_dim'] = 512
        
        # 파이프라인 생성
        print(f"🏗️ Building dynamic pipeline...")
        print(f"   📝 Text encoder: {encoder_type}")
        print(f"   🏗️ Backbone: {backbone_type}")
        print(f"   🎵 CLAP enabled: {use_clap}")
        
        pipeline = TextToAudioProcessingPipeline(
            text_encoder_type=encoder_type,
            text_encoder_config=encoder_config,
            use_clap=use_clap,
            backbone_type=backbone_type,
            backbone_config=backbone_config,
            **pipeline_kwargs
        )
        
        return pipeline
    
    @classmethod
    def list_presets(cls):
        """사용 가능한 인코더 프리셋 목록 출력"""
        print("🎛️ Available Encoder Presets:")
        print("=" * 60)
        
        for name, preset in cls.ENCODER_PRESETS.items():
            print(f"📌 {name}")
            print(f"   Type: {preset['type']}")
            print(f"   Description: {preset['description']}")
            print(f"   Config: {preset['config']}")
            print()
    
    @classmethod
    def benchmark_encoders(cls, test_texts: list = None, device: str = 'cuda'):
        """다양한 인코더의 성능을 벤치마크"""
        if test_texts is None:
            test_texts = [
                "add heavy distortion and reverb",
                "make it sound warmer with chorus", 
                "apply vintage analog compression",
                "create spacious ambient echo"
            ]
        
        print("🔬 Encoder Performance Benchmark")
        print("=" * 60)
        
        results = {}
        
        for preset_name in cls.ENCODER_PRESETS.keys():
            try:
                print(f"\n🧪 Testing {preset_name}...")
                
                # 파이프라인 생성
                pipeline = cls.create_pipeline(
                    encoder_preset=preset_name,
                    use_clap=False  # 순수 텍스트 인코더만 테스트
                )
                pipeline = pipeline.to(device)
                
                # 성능 측정
                import time
                
                # Warm up
                for _ in range(3):
                    _ = pipeline.encode_text(test_texts)
                
                # 실제 측정
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                for _ in range(10):  # 10회 평균
                    embeddings = pipeline.encode_text(test_texts)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10 * 1000  # ms
                per_item_time = avg_time / len(test_texts)
                
                # 메모리 사용량
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated() / 1024**2
                else:
                    memory_mb = 0
                
                results[preset_name] = {
                    'avg_time_ms': avg_time,
                    'per_item_ms': per_item_time,
                    'memory_mb': memory_mb,
                    'embedding_dim': embeddings[0].shape[-1] if isinstance(embeddings, tuple) else embeddings.shape[-1],
                    'success': True
                }
                
                print(f"   ✅ Success - {avg_time:.2f}ms total, {per_item_time:.2f}ms/item")
                print(f"      Memory: {memory_mb:.1f}MB, Dim: {results[preset_name]['embedding_dim']}")
                
                # 메모리 정리
                del pipeline
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[preset_name] = {'success': False, 'error': str(e)}
        
        # 결과 요약
        print(f"\n📊 Benchmark Summary:")
        print("-" * 60)
        print(f"{'Preset':<25} {'Time(ms)':<10} {'Per Item':<10} {'Memory(MB)':<12} {'Dim':<6}")
        print("-" * 60)
        
        for name, result in results.items():
            if result['success']:
                print(f"{name:<25} {result['avg_time_ms']:<10.2f} "
                      f"{result['per_item_ms']:<10.2f} {result['memory_mb']:<12.1f} "
                      f"{result['embedding_dim']:<6}")
            else:
                print(f"{name:<25} {'FAILED':<10} {'N/A':<10} {'N/A':<12} {'N/A':<6}")
        
        return results
    
    @classmethod 
    def quick_test(cls, encoder_preset: str = 'sentence-transformer-large',
                   backbone_type: str = 'simple',
                   test_text: str = "add heavy reverb and distortion"):
        """빠른 테스트를 위한 헬퍼 함수"""
        print(f"🚀 Quick test with {encoder_preset} + {backbone_type} backbone")
        
        pipeline = cls.create_pipeline(
            encoder_preset=encoder_preset,
            backbone_type=backbone_type
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline = pipeline.to(device)
        
        # 텍스트 인코딩 테스트
        embeddings = pipeline.encode_text([test_text])
        
        print(f"✅ Success!")
        print(f"   Input: '{test_text}'")
        if isinstance(embeddings, tuple):
            text_emb, clap_emb = embeddings
            print(f"   Text embedding: {text_emb.shape}")
            if clap_emb is not None:
                print(f"   CLAP embedding: {clap_emb.shape}")
        else:
            print(f"   Embedding shape: {embeddings.shape}")
        
        return pipeline


if __name__ == "__main__":
    # 사용 예시
    factory = DynamicPipelineFactory()
    
    # 프리셋 목록 출력
    factory.list_presets()
    
    # 빠른 테스트
    print("\n" + "="*60)
    pipeline = factory.quick_test('sentence-transformer-large', 'dynamic')  # 동적 백본으로 테스트
    
    # 성능 벤치마크 (옵션)
    if input("\n벤치마크를 실행하시겠습니까? (y/n): ").lower() == 'y':
        factory.benchmark_encoders()
