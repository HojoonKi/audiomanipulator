#!/usr/bin/env python3
"""
another.txt에서 딕셔너리들을 추출해서 fined_presets.txt에 추가하는 스크립트
"""

import json
import re

def extract_dicts_from_another():
    """another.txt에서 딕셔너리들을 추출"""
    
    with open('/workspace/AudioManipulator/descriptions/another.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📖 another.txt 파일 크기: {len(content)} characters")
    
    # JSON 배열들을 찾기
    extracted_dicts = []
    
    # [ ... ] 블록들을 찾기
    bracket_blocks = []
    i = 0
    while i < len(content):
        if content[i] == '[':
            start = i
            bracket_count = 1
            i += 1
            
            while i < len(content) and bracket_count > 0:
                if content[i] == '[':
                    bracket_count += 1
                elif content[i] == ']':
                    bracket_count -= 1
                i += 1
            
            if bracket_count == 0:
                block = content[start:i]
                bracket_blocks.append(block)
        else:
            i += 1
    
    print(f"🔍 발견된 [] 블록 수: {len(bracket_blocks)}")
    
    # 각 블록에서 딕셔너리 추출
    for block_idx, block in enumerate(bracket_blocks):
        try:
            # JSON으로 파싱 시도
            parsed = json.loads(block)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and 'prompt' in item:
                        extracted_dicts.append(item)
                        print(f"   추출: {item['prompt'][:50]}...")
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 수동으로 딕셔너리 찾기
            print(f"⚠️ 블록 {block_idx}: JSON 파싱 실패, 수동 추출 시도")
            
            # { ... } 딕셔너리 블록들 찾기
            dict_blocks = []
            j = 0
            while j < len(block):
                if block[j] == '{':
                    start = j
                    brace_count = 1
                    j += 1
                    
                    while j < len(block) and brace_count > 0:
                        if block[j] == '{':
                            brace_count += 1
                        elif block[j] == '}':
                            brace_count -= 1
                        j += 1
                    
                    if brace_count == 0:
                        dict_block = block[start:j]
                        dict_blocks.append(dict_block)
                else:
                    j += 1
            
            # 각 딕셔너리 블록을 파싱
            for dict_block in dict_blocks:
                try:
                    # 주석 제거
                    cleaned = re.sub(r'//.*?$', '', dict_block, flags=re.MULTILINE)
                    cleaned = re.sub(r'#.*?$', '', cleaned, flags=re.MULTILINE)
                    
                    # JSON으로 파싱
                    parsed_dict = json.loads(cleaned)
                    if isinstance(parsed_dict, dict) and 'prompt' in parsed_dict:
                        extracted_dicts.append(parsed_dict)
                        print(f"   수동 추출: {parsed_dict['prompt'][:50]}...")
                except:
                    continue
    
    print(f"✅ 총 추출된 딕셔너리 수: {len(extracted_dicts)}")
    return extracted_dicts

def append_to_fined_presets(new_dicts):
    """fined_presets.txt에 새로운 딕셔너리들 추가"""
    
    with open('/workspace/AudioManipulator/descriptions/fined_presets.txt', 'a', encoding='utf-8') as f:
        f.write('\n\n# --- 추가된 presets from another.txt ---\n\n')
        
        for i, preset_dict in enumerate(new_dicts):
            # 딕셔너리를 JSON 형태로 출력 (읽기 쉽게 정렬)
            json_str = json.dumps(preset_dict, indent=2, ensure_ascii=False)
            f.write(json_str)
            f.write('\n\n')
            
            if (i + 1) % 10 == 0:
                print(f"📝 {i + 1}개 딕셔너리 추가됨...")
    
    print(f"✅ 총 {len(new_dicts)}개 딕셔너리가 fined_presets.txt에 추가되었습니다!")

def main():
    """메인 함수"""
    print("🚀 another.txt에서 딕셔너리 추출 시작")
    
    # 딕셔너리들 추출
    extracted_dicts = extract_dicts_from_another()
    
    if not extracted_dicts:
        print("❌ 추출된 딕셔너리가 없습니다.")
        return
    
    # fined_presets.txt에 추가
    append_to_fined_presets(extracted_dicts)
    
    print("🎉 작업 완료!")
    
    # 통계 출력
    print(f"\n📊 추출 통계:")
    print(f"   - 총 딕셔너리 수: {len(extracted_dicts)}")
    
    # 프롬프트 키워드 분석
    keywords = {}
    for preset in extracted_dicts:
        prompt = preset.get('prompt', '').lower()
        words = prompt.split()
        for word in words:
            if len(word) > 3:  # 3글자 이상 단어만
                keywords[word] = keywords.get(word, 0) + 1
    
    # 상위 키워드 출력
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"   - 상위 키워드: {[f'{k}({v})' for k, v in sorted_keywords]}")

if __name__ == "__main__":
    main()
