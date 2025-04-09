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
==================================================================================================

def contains_nearby_duplicate(nums: list[int], k: int) -> bool:
    b = {}
    for i, num in enumerate(nums):
        if num in b and i - b[num] <= k:
            return True
        b[num] = i
    return False
    

# 🔍 Test cases to validate your solution
if __name__ == "__main__":
    test_cases = [
        ([1, 2, 3, 1], 3, True),
        ([1, 0, 1, 1], 1, True),
        ([1, 2, 3, 1, 2, 3], 2, False),
        ([99, 99], 2, True),
        ([1, 2, 3, 4, 5, 6], 0, False),
        ([1], 1, False)
    ]

    for i, (nums, k, expected) in enumerate(test_cases):
        result = contains_nearby_duplicate(nums, k)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

===================================================================================================

def is_valid_parentheses(s: str) -> bool:
    p_stack = []
    for i in range(len(s)):
        if(s[i] == "(" or s[i] == "[" or s[i] == "{"):
            p_stack.append(s[i])
            
        else:
            if (p_stack and
            ((p_stack[-1] == "(" and s[i] == ")") or
            (p_stack[-1] == "[" and s[i] == "]") or
            (p_stack[-1] == "{" and s[i] == "}"))):
                p_stack.pop()
            else:
                return False
    return not p_stack

Another solution:
def is_valid_parentheses(s: str) -> bool:
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    
    for char in s:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            return False  # invalid character
    return not stack


if __name__ == "__main__":
    test_cases = [
        ("()", True),
        ("()[]{}", True),
        ("(]", False),
        ("([)]", False),
        ("{[]}", True),
        ("", True),
        ("[", False),
        ("([])", True)
    ]

    for i, (inp, expected) in enumerate(test_cases):
        result = is_valid_parentheses(inp)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

=====================================================================================================


from collections import deque

def num_islands(grid: list[list[str]]) -> int:
    if not grid:
        return 0
        
    rows, cols = len(grid), len(grid[0])
    visit = set()
    islands = 0
    
    def bfs(r,c):
        q = deque()
        visit.add((r,c))
        q.append((r,c))
        
        while q:
            row, col = q.popleft()
            directions = [[1,0], [-1,0], [0,1], [0,-1]]
            
            for dr, dc in directions:
                new_r, new_c = row + dr, col + dc
                if (new_r in range(rows) and 
                new_c in range(cols) and grid[new_r][new_c] == "1"
                and (new_r,new_c) not in visit):
                    q.append((new_r, new_c))
                    visit.add((new_r, new_c))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r,c) not in visit:
                bfs(r,c)
                islands = islands +1
    return islands
    
    
            

if __name__ == "__main__":
    test_grids = [
        (
            [["1","1","0","0","0"],
             ["1","1","0","0","0"],
             ["0","0","1","0","0"],
             ["0","0","0","1","1"]],
            3
        ),
        (
            [["1","1","1"],
             ["0","1","0"],
             ["1","1","1"]],
            1
        ),
        (
            [["0","0","0"],
             ["0","0","0"],
             ["0","0","0"]],
            0
        ),
        (
            [["1"]],
            1
        ),
        (
            [["0"]],
            0
        )
    ]

    for i, (grid, expected) in enumerate(test_grids):
        result = num_islands(grid)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

=============================================================================================

from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order(root: TreeNode) -> list[list[int]]:
    if not root:
        return 0
        
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
        
    return result

# Example tree:
#         3
#        / \
#       9  20
#          / \
#         15  7
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

print("Output:", level_order(root))  # Expected: [[3], [9, 20], [15, 7]]

====================================================================================

def majority_element(nums: list[int]) -> int:
    d = {}
    count_nums = []
    division = len(nums)//2
    nums_set = set(nums)
    for i in nums_set:
        count = nums.count(i)
        count_nums.append(count)
    res = dict(zip(nums_set, count_nums))
    result = max(res, key=res.get)
    
    return result
        
    

if __name__ == "__main__":
    test_cases = [
        ([3, 2, 3], 3),
        ([2, 2, 1, 1, 1, 2, 2], 2),
        ([1, 1, 1, 2, 3, 1], 1),
        ([4, 4, 5], 4)
    ]

    for i, (nums, expected) in enumerate(test_cases):
        result = majority_element(nums)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

===================================================================================

def backspace_compare(s: str, t: str) -> bool:
    def process(string):
        t1 = []
        for i in string:
            if(i != "#"):
                t1.append(i)
            elif t1:
                t1.pop()
        return t1
            
    return process(s) == process(t)

if __name__ == "__main__":
    test_cases = [
        ("ab#c", "ad#c", True),     # both become "ac"
        ("ab##", "c#d#", True),     # both become ""
        ("a#c", "b", False),
        ("xy#z", "xzz#", True),     # both become "xz"
        ("a##c", "#a#c", True)
    ]

    for i, (s, t, expected) in enumerate(test_cases):
        result = backspace_compare(s, t)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

=====================================================================================

def min_subarray_len(target: int, nums: list[int]) -> int:
    left = 0
    current_sum = 0
    min_len = float('inf')
    for right in range(len(nums)):
        current_sum += nums[right]
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= nums[left]
            left = left +1
    if min_len == float('inf'):
        return 0
    else:
        return min_len
        

if __name__ == "__main__":
    test_cases = [
        (7, [2,3,1,2,4,3], 2),        # [4,3]
        (4, [1,4,4], 1),              # [4]
        (11, [1,1,1,1,1,1,1,1], 0),   # no subarray ≥ 11
        (15, [1,2,3,4,5], 5),         # entire array
        (5, [2,3], 2)                 # [2,3]
    ]

    for i, (target, nums, expected) in enumerate(test_cases):
        result = min_subarray_len(target, nums)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

=============================================================================================

from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def has_path_sum(root: TreeNode, target_sum: int) -> bool:
    if not root:
        return 0
        
    result = 0
    queue = deque([(root, root.val)])
    
    while queue:
        for _ in range(len(queue)):
            node, curr_sum = queue.popleft()
            if not node.left and not node.right and curr_sum == target_sum:
                return True
            if node.left:
                queue.append((node.left, curr_sum + node.left.val))
            if node.right:
                queue.append((node.right, curr_sum + node.right.val))

    return False

        
    

if __name__ == "__main__":
    # Tree from example
    root = TreeNode(5)
    root.left = TreeNode(4)
    root.right = TreeNode(8)
    root.left.left = TreeNode(11)
    root.left.left.left = TreeNode(7)
    root.left.left.right = TreeNode(2)
    root.right.left = TreeNode(13)
    root.right.right = TreeNode(4)
    root.right.right.right = TreeNode(1)

    print("Expected: True | Output:", has_path_sum(root, 22))

================================================================================================

import string

def check_if_pangram(sentence: str) -> bool:
    return set(string.ascii_lowercase) <= set(sentence.lower())

if __name__ == "__main__":
    test_cases = [
        ("thequickbrownfoxjumpsoverthelazydog", True),
        ("leetcode", False),
        ("abcdefghijklmnopqrstuvwxyz", True),
        ("", False),
        ("abcdefghijklmnopqrstuvwxy", False)
    ]

    for i, (s, expected) in enumerate(test_cases):
        result = check_if_pangram(s)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

===================================================================================================

def remove_duplicates(s: str) -> str:
    new_stack = []
    for i in s:
        if new_stack and i == new_stack[-1]:
            new_stack.pop()
        else:
            new_stack.append(i)
        
    return ''.join(new_stack)
            

if __name__ == "__main__":
    test_cases = [
        ("abbaca", "ca"),
        ("azxxzy", "ay"),
        ("a", "a"),
        ("aa", ""),
        ("aabbcc", "")
    ]

    for i, (s, expected) in enumerate(test_cases):
        result = remove_duplicates(s)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

======================================================================================================

from collections import defaultdict

def length_of_longest_substring_two_distinct(s: str) -> int:
    count = 0
    left = 0
    a = defaultdict(int)
    if len(s) <= 2:
        return len(s)
    for right in range(len(s)):
        a[s[right]] += 1
        
        while len(a) > 2:
            a[s[left]] -= 1
            if a[s[left]] == 0:
                del a[s[left]]
            left += 1

        count = max(count, right -left +1)
    return count
            

if __name__ == "__main__":
    test_cases = [
        ("eceba", 3),         # "ece"
        ("ccaabbb", 5),       # "aabbb"
        ("a", 1),
        ("ab", 2),
        ("abcabcabc", 2)
    ]

    for i, (s, expected) in enumerate(test_cases):
        result = length_of_longest_substring_two_distinct(s)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

==============================================================================================

from collections import Counter
import heapq

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    freq = Counter(nums)
    
    max_heap = []
    for num, count in freq.items():
        heapq.heappush(max_heap, (-count, num))
        
    result = []
    for _ in range(k):
        count, num = heapq.heappop(max_heap)
        result.append(num)
        
    return result

if __name__ == "__main__":
    test_cases = [
        ([1,1,1,2,2,3], 2, [1,2]),
        ([1], 1, [1]),
        ([4,1,-1,2,-1,2,3], 2, [-1, 2]),
        ([5,5,5,6,6,7], 1, [5]),
        ([0,0,0,1,2,2,3,3,3,3], 2, [3,0])
    ]

    for i, (nums, k, expected) in enumerate(test_cases):
        result = top_k_frequent(nums, k)
        print(f"Test {i+1}: Output: {result} | Expected Top {k}: {expected}")

===================================================================================================

from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def good_nodes(root: TreeNode) -> int:
    if not root:
        return 0
    
    count = 0
    queue = deque([(root, root.val)])
    
    while queue:
            node, max_val = queue.popleft()
            
            if node.val >= max_val:
                count = count + 1
            new_max = max(max_val, node.val)
            if node.left:
                queue.append((node.left, new_max))
            if node.right:
                queue.append((node.right, new_max))
    return count
            

if __name__ == "__main__":
    # Build the example tree
    root = TreeNode(3)
    root.left = TreeNode(1)
    root.right = TreeNode(4)
    root.left.left = TreeNode(3)
    root.right.left = TreeNode(1)
    root.right.right = TreeNode(5)

    print("Expected: 4 | Output:", good_nodes(root))

==========================================================================================

def subarray_sum(nums: list[int], k: int) -> int:
    count = 0
    for i in range(len(nums)):
        sum = 0
        j = i
        while(j< len(nums)):
            sum += nums[j]
            if(sum == k):
                count += 1
            j += 1
    return count

if __name__ == "__main__":
    test_cases = [
        ([1, 1, 1], 2, 2),
        ([1, 2, 3], 3, 2),          # [3], [1,2]
        ([1], 0, 0),
        ([0,0,0,0,0], 0, 15),       # All subarrays sum to 0
        ([1, -1, 0], 0, 3)
    ]

    for i, (nums, k, expected) in enumerate(test_cases):
        result = subarray_sum(nums, k)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

============================================================================================

def search_range(nums: list[int], target: int) -> list[int]:
    l = []
    if target not in nums:
        return [-1, -1]
    else:
        for i in range(len(nums)):
            if nums[i] == target:
                l.append(i)
    return [l[0], l[-1]]
        

if __name__ == "__main__":
    test_cases = [
        ([5,7,7,8,8,10], 8, [3, 4]),
        ([5,7,7,8,8,10], 6, [-1, -1]),
        ([], 0, [-1, -1]),
        ([1], 1, [0, 0]),
        ([2,2], 2, [0, 1])
    ]

    for i, (nums, target, expected) in enumerate(test_cases):
        result = search_range(nums, target)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

==============================================================================================

Given an array of non-negative integers nums, where each element represents your maximum jump length, determine if you can reach the last index starting from index 0.

def can_jump(nums: list[int]) -> bool:
    max_reach = 0
    
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
        
    return True

if __name__ == "__main__":
    test_cases = [
        ([2,3,1,1,4], True),
        ([3,2,1,0,4], False),
        ([0], True),
        ([2,0,0], True),
        ([1,1,0,1], False)
    ]

    for i, (nums, expected) in enumerate(test_cases):
        result = can_jump(nums)
        print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'} | Output: {result} | Expected: {expected}")

==============================================================================================

def min_remove_to_make_valid(s: str) -> str:
    stack = []
    remove = set()
    
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                stack.pop()
            else:
                remove.add(i)
                
    remove.update(stack)
    
    result = ""
    for i, ch in enumerate(s):
        if i not in remove:
            result +=ch
            
    return result

if __name__ == "__main__":
    test_cases = [
        ("lee(t(c)o)de)", "lee(t(c)o)de"),
        ("a)b(c)d", "ab(c)d"),
        ("))((", ""),
        ("(a(b(c)d)", "a(b(c)d)"),
        ("a(b))c(d)", "a(b)c(d)")
    ]

    for i, (input_s, expected) in enumerate(test_cases):
        result = min_remove_to_make_valid(input_s)
        print(f"Test {i+1}: Output = '{result}' | Expected = '{expected}'")

===========================================================================================

from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def sum_of_left_leaves(root: TreeNode) -> int:
    if not root:
        return 0
        
    queue = deque([root])
    total = 0
    
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                if not node.left.left and not node.left.right:
                    total += node.left.val
                else:
                    queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return total
    

if __name__ == "__main__":
    # Example test case
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20, TreeNode(15), TreeNode(7))
    
    print("Sum of Left Leaves:", sum_of_left_leaves(root))  # Expected: 24

==============================================================================================

import re
from collections import Counter

def most_common_word(paragraph: str, banned: list[str]) -> str:
    clean = re.sub(r'[^\w\s]', '', paragraph)
    splited_para = clean.lower().split()
    freq = Counter(splited_para)
    
    for i in banned:
        if i in freq:
            del freq[i]
            
    return max(freq, key=freq.get)

    
    

if __name__ == "__main__":
    p = "Bob hit a ball, the hit BALL flew far after it was hit."
    banned = ["hit"]
    print("Most Common:", most_common_word(p, banned))  # Expected: "ball"

==============================================================================================

def can_be_typed_words(text: str, broken_letters: str) -> int:
    count = 0
    text_list = text.split()
    bl_list = set(broken_letters)
    for i in text_list:
        for j in i:
            if j in bl_list:
                break
        else:
            count += 1
    return count

if __name__ == "__main__":
    print(can_be_typed_words("hello world", "ad"))        # → 1
    print(can_be_typed_words("leet code", "lt"))          # → 1
    print(can_be_typed_words("leet code", "e"))           # → 0

=============================================================================================

from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def count_leaves(root: TreeNode) -> int:
    if not root:
        return 0
        
    count = 0
    queue = deque([root])
    
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            if not node.left and not node.right:
                count += 1
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
                
    return count
            

if __name__ == "__main__":
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20, TreeNode(15), TreeNode(7))

    print("Leaf count:", count_leaves(root))  # Expected: 3

===================================================================================

def longest_common_prefix(strs: list[str]) -> str:
    prefix = strs[0]
    
    while(len(prefix)>0):
        for i in range(1, len(strs)):
            if all(strs[i].startswith(prefix) for word in strs):
                return prefix
        prefix = prefix[:len(prefix)-1]
        
    return ""
        
            

if __name__ == "__main__":
    print(longest_common_prefix(["flower","flow","flight"]))  # → "fl"
    print(longest_common_prefix(["dog","racecar","car"]))     # → ""
    print(longest_common_prefix(["interspecies","interstellar","interstate"]))  # → "interest"

====================================================================================

from collections import Counter

def num_identical_pairs(nums: list[int]) -> int:
    count = 0
    for i in range(len(nums)):
        j = i +1
        while(j < len(nums)):
            if(nums[i] == nums[j] and i<j):
                count += 1
            j += 1
    return count
        

if __name__ == "__main__":
    print(num_identical_pairs([1,2,3,1,1,3]))  # → 4
    print(num_identical_pairs([1,1,1,1]))      # → 6
    print(num_identical_pairs([1,2,3]))        # → 0

====================================================================================

from collections import Counter

def find_the_difference(s: str, t: str) -> str:
    diff = Counter(t) - Counter(s)
    return next(iter(diff))

if __name__ == "__main__":
    print(find_the_difference("abcd", "abcde"))  # → "e"
    print(find_the_difference("", "y"))          # → "y"
    print(find_the_difference("aabbcc", "abcbcad"))  # → "d"

