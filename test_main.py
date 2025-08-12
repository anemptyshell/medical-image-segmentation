import sys
from typing import List, Set, Tuple

def findSubarrays(nums: List[int], target: int) -> List[List[int]]:
    result = []
    current = []
    
    def backtrack(start, current_sum):
        # 找到符合条件的子数组
        if current_sum == target and current:
            result.append(current)
            return
        # 超过目标值或遍历完数组
        if current_sum > target or start >= len(nums):
            return
        
        # 选择当前元素
        current.append(nums[start])
        backtrack(start + 1, current_sum + nums[start])
        current.pop()  # 回溯
        
        # 不选择当前元素
        backtrack(start + 1, current_sum)
    
    backtrack(0, 0)
    return result

# 去重处理（若有重复元素）
def deduplicate(results: List[List[int]]) -> List[List[int]]:
    # 将子数组转为元组后去重
    return [list(t) for t in set(tuple(sub) for sub in results)]

# ACM模式输入处理
def main():
    # 读取第一行：数组长度n和目标值target
    line1 = sys.stdin.readline().strip()
    n, target = map(int, line1.split())
    
    # 读取第二行：数组元素
    line2 = sys.stdin.readline().strip()
    nums = list(map(int, line2.split()))
    
    # 求解并去重
    results = findSubarrays(nums, target)
    unique_results = deduplicate(results)
    
    # 输出结果
    for subarray in unique_results:
        print(' '.join(map(str, subarray)))

if __name__ == "__main__":
    main()