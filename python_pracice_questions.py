def length_of_longest_substring(s: str) -> int:
    a = set()
    left = 0
    substring_count = 0
    for right in range(len(s)):
        while s[right] in a:
            a.remove(s[left])
            left = left + 1
        a.add(s[right])
        substring_count = max(substring_count, right -left +1)
    return substring_count
