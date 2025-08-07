#!/usr/bin/env python3
"""
another.txt에서 딕셔너리들을 원본 포맷 그대로 추출해서 fined_presets.txt에 추가
주석과 줄바꿈을 모두 보존
"""

def extract_and_append_raw():
    """원본 포맷을 그대로 유지하면서 딕셔너리 추출"""
    
    with open('/workspace/AudioManipulator/descriptions/another.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📖 another.txt 파일 크기: {len(content):,} characters")
    
    # [ ... ] 블록들을 찾기 (원본 포맷 그대로)
    extracted_blocks = []
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
                # 빈 블록이 아닌지 확인 (실제 딕셔너리가 있는지)
                if '{' in block and '"prompt"' in block:
                    extracted_blocks.append(block)
        else:
            i += 1
    
    print(f"🔍 발견된 유효한 [] 블록 수: {len(extracted_blocks)}")
    
    # 각 블록에서 딕셔너리 개수 확인
    total_dicts = 0
    for block_idx, block in enumerate(extracted_blocks):
        dict_count = block.count('"prompt"')
        total_dicts += dict_count
        print(f"   블록 {block_idx+1}: {dict_count}개 딕셔너리")
    
    print(f"📊 총 예상 딕셔너리 수: {total_dicts}")
    
    # fined_presets.txt에 블록별로 추가
    with open('/workspace/AudioManipulator/descriptions/fined_presets.txt', 'a', encoding='utf-8') as f:
        f.write('\n\n# ===== 추가된 presets from another.txt =====\n')
        f.write('# 원본 포맷 그대로 보존 (주석 및 줄바꿈 포함)\n\n')
        
        for block_idx, block in enumerate(extracted_blocks):
            f.write(f'# --- Block {block_idx+1} ---\n')
            
            # 블록에서 개별 딕셔너리들 추출 (원본 포맷 유지)
            # [ ] 를 제거하고 내부 딕셔너리들만 추출
            block_content = block.strip()
            if block_content.startswith('[') and block_content.endswith(']'):
                # 대괄호 제거
                inner_content = block_content[1:-1].strip()
                
                # 개별 딕셔너리들을 분리 (} 다음에 ,가 오는 부분을 기준으로)
                dict_parts = []
                current_dict = ""
                brace_count = 0
                in_string = False
                escape_next = False
                
                for char in inner_content:
                    if escape_next:
                        current_dict += char
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        current_dict += char
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                    
                    current_dict += char
                    
                    # 딕셔너리가 완료되었는지 확인
                    if not in_string and brace_count == 0 and current_dict.strip():
                        # 다음 문자가 ','인지 확인하고 딕셔너리 완료
                        if current_dict.strip().endswith('}'):
                            dict_parts.append(current_dict.strip().rstrip(','))
                            current_dict = ""
                
                # 마지막 딕셔너리 처리
                if current_dict.strip():
                    dict_parts.append(current_dict.strip().rstrip(','))
                
                # 각 딕셔너리를 개별적으로 추가
                for dict_idx, dict_content in enumerate(dict_parts):
                    if dict_content.strip() and '{' in dict_content:
                        f.write(dict_content)
                        f.write('\n\n')
                
                print(f"   블록 {block_idx+1}: {len(dict_parts)}개 딕셔너리 추가됨")
    
    print(f"✅ 모든 딕셔너리가 원본 포맷 그대로 fined_presets.txt에 추가되었습니다!")
    return total_dicts

def verify_result():
    """결과 검증"""
    with open('/workspace/AudioManipulator/descriptions/fined_presets.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # "prompt" 개수로 딕셔너리 수 확인
    prompt_count = content.count('"prompt"')
    print(f"📋 fined_presets.txt의 총 딕셔너리 수: {prompt_count}")
    
    # 파일 크기 확인
    file_size = len(content)
    print(f"📁 fined_presets.txt 파일 크기: {file_size:,} characters")

def main():
    """메인 함수"""
    print("🚀 another.txt에서 원본 포맷 그대로 딕셔너리 추출 시작")
    
    # 기존 fined_presets.txt 백업
    import shutil
    try:
        shutil.copy('/workspace/AudioManipulator/descriptions/fined_presets.txt', 
                   '/workspace/AudioManipulator/descriptions/fined_presets_backup.txt')
        print("💾 기존 fined_presets.txt 백업 완료")
    except:
        print("ℹ️  백업 파일 생성 건너뜀")
    
    # 딕셔너리들 추출 및 추가
    total_extracted = extract_and_append_raw()
    
    # 결과 검증
    verify_result()
    
    print("🎉 작업 완료!")

if __name__ == "__main__":
    main()
