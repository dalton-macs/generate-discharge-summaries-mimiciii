import re

def get_discharge_headers(text):
    
    exclusion_headers = ['admission date', 'discharge date', 'date of birth', 'sex', 'service']
    
    pattern = r'\n\n([^0-9].*?):[\n\s+]'
    matches = re.findall(pattern, text, re.DOTALL)
    match_list = []
    
    for match in matches:
        curr_match = None
        # print(match)
        # print(len(match))
        if len(match)<40:
            index = text.find(f"\n\n{match}:")+len(match)
            # print(text[index:index+10])
            if '[' not in text[index:index+10]:
            # print(match)
                curr_match = match.replace('\n', '')
                
        else:
            # print('trying by implementing heuristics')
            substring = "\n\n"
            index = match.rfind(substring)  # Find the last occurrence of the substring
            
            if index != -1:
                result = match[index+len(substring):]  # Extract the substring from the last occurrence to the end
                # print(result)
                index = text.find(f"\n\n{result}:")+len(result)
                # print(text[index:index+10])
                if '[' not in text[index:index+10] and not re.match(r'^\d', result):
                    curr_match = result.replace('\n', '')

        if (curr_match) and (curr_match.lower() not in exclusion_headers) and (len(curr_match)<50):
            match_list.append(curr_match.upper())
    
    return match_list


def get_target_text(text, headers, target_headers = ['BRIEF HOSPITAL COURSE', 'HOSPITAL COURSE']):

    if isinstance(target_headers, str):
        target_headers = [target_headers]
    
    text_dummy = text.lower()

    found_counter = 0
    for header in target_headers:
        if header in headers:
            target_header = header
            found_counter += 1

    if found_counter == 0:
        return "NOT FOUND"

    elif found_counter > 1:
        return "MULTIPLE FOUND"

    else:
        if headers.index(target_header) == len(headers)-1:
            end_idx = len(text)
        else:
            next_header = headers[headers.index(target_header)+1].lower()
            end_idx = text_dummy.find(next_header)
            
        target_header = target_header.lower()
        start_idx = text_dummy.find(target_header)+len(target_header)
        
    
        target_text = text[start_idx:end_idx].lstrip(':').strip().replace('\n', ' ')
        return target_text