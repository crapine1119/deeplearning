import numpy as np

s1 = "kitten"
s2 = "sitting"


len_s1, len_s2 = len(s1), len(s2)
# 편집 거리 행렬 초기화
dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)

# 첫 번째 문자열의 각 문자를 공백으로 바꾸는 비용
for i in range(len_s1 + 1):
    dp[i][0] = i

# 두 번째 문자열의 각 문자를 공백으로 바꾸는 비용
for j in range(len_s2 + 1):
    dp[0][j] = j

# 편집 거리 계산
for i in range(1, len_s1 + 1):
    for j in range(1, len_s2 + 1):
        cost = 0 if s1[i - 1] == s2[j - 1] else 1
        dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

edit_dist = dp[-1][-1]

from nltk import edit_distance

edit_distance(s1, s2)
