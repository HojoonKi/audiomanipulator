#!/usr/bin/env python3
"""
another.txt의 딕셔너리 개수를 정확히 세는 스크립트
"""

def count_dictionaries():
    """딕셔너리 개수 정확히 카운트"""
    
    with open('/workspace/AudioManipulator/descriptions/another.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📖 another.txt 파일 크기: {len(content):,} characters")
    print(f"📖 총 라인 수: {content.count(chr(10)) + 1:,} lines")
    
    # 1. "prompt" 개수로 딕셔너리 개수 추정
    prompt_count = content.count('"prompt"')
    print(f"🔍 \"prompt\" 키워드 개수: {prompt_count}")
    
    # 2. 중괄호 블록 개수 세기
    brace_blocks = 0
    i = 0
    while i < len(content):
        if content[i] == '{':
            brace_count = 1
            i += 1
            while i < len(content) and brace_count > 0:
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                i += 1
            if brace_count == 0:
                brace_blocks += 1
        else:
            i += 1
    
    print(f"🔍 중괄호 {{}} 블록 개수: {brace_blocks}")
    
    # 3. 대괄호 블록 개수와 내용 분석
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
    
    print(f"🔍 대괄호 [] 블록 개수: {len(bracket_blocks)}")
    
    # 각 대괄호 블록 분석
    total_dict_count = 0
    for idx, block in enumerate(bracket_blocks):
        dict_count_in_block = block.count('"prompt"')
        total_dict_count += dict_count_in_block
        
        # 블록 크기도 확인
        block_size = len(block)
        print(f"   블록 {idx+1}: {dict_count_in_block}개 딕셔너리, {block_size:,} characters")
        
        # 첫 몇 글자 미리보기
        preview = block[:100].replace('\n', ' ').replace('  ', ' ')
        print(f"      미리보기: {preview}...")
    
    print(f"\n📊 최종 통계:")
    print(f"   - 대괄호 블록에서 찾은 총 딕셔너리: {total_dict_count}")
    print(f"   - 전체 파일에서 \"prompt\" 개수: {prompt_count}")
    print(f"   - 전체 파일에서 중괄호 블록 개수: {brace_blocks}")
    
    return total_dict_count

def analyze_file_structure():
    """파일 구조 더 자세히 분석"""
    
    with open('/workspace/AudioManipulator/descriptions/another.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\n📋 파일 구조 분석:")
    print(f"   - 총 라인 수: {len(lines)}")
    
    # 각 라인 타입 분석
    bracket_open_lines = []
    bracket_close_lines = []
    prompt_lines = []
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == '[':
            bracket_open_lines.append(line_num)
        elif stripped == ']':
            bracket_close_lines.append(line_num)
        elif '"prompt"' in stripped:
            prompt_lines.append(line_num)
    
    print(f"   - '[' 라인들: {bracket_open_lines}")
    print(f"   - ']' 라인들: {bracket_close_lines}")
    print(f"   - \"prompt\" 포함 라인 수: {len(prompt_lines)}")
    
    # 각 블록의 라인 범위 계산
    if len(bracket_open_lines) == len(bracket_close_lines):
        print(f"\n📦 블록별 라인 범위:")
        for i in range(len(bracket_open_lines)):
            start_line = bracket_open_lines[i]
            end_line = bracket_close_lines[i]
            block_lines = end_line - start_line + 1
            
            # 해당 블록 내의 prompt 개수
            prompts_in_block = sum(1 for line_num in prompt_lines 
                                 if start_line <= line_num <= end_line)
            
            print(f"   블록 {i+1}: 라인 {start_line}-{end_line} ({block_lines} 라인, {prompts_in_block} prompts)")

def main():
    print("🔍 another.txt 딕셔너리 개수 정확히 세기")
    
    count = count_dictionaries()
    analyze_file_structure()
    
    print(f"\n🎯 결론: 총 {count}개의 딕셔너리가 발견되었습니다!")

if __name__ == "__main__":
    main()
