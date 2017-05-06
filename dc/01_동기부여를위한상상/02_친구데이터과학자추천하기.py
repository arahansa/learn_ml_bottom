##########################
#                        #
#      친구끼리 커넥션 찾기    #
#                        #
##########################

users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" },
    { "id": 10, "name": "Jen" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]


# first give each user an empty list
for user in users:
    user["friends"] = []

# and then populate the lists with friendships
for i, j in friendships:
    # this works because users[i] is the user whose id is i
    users[i]["friends"].append(users[j]) # add i as a friend of j
    users[j]["friends"].append(users[i]) # add j as a friend of i

def number_of_friends(user):
    """how many friends does _user_ have?"""
    return len(user["friends"]) # length of friend_ids list

total_connections = sum(number_of_friends(user) for user in users) # 24

num_users = len(users)
avg_connections = total_connections / num_users # 2.4

print("전체 커넥션", total_connections)
print("num_users : ", num_users)
print("평균 커넥션 : ", avg_connections)

################################
#                              #
# 데이터 과학자 추천하기 #
#                              #
################################

def friends_of_friend_ids_bad(user):
    return [foaf["id"]
            for friend in user["friends"]
            for foaf in friend["friends"] ]

for u in users[0]["friends"]:
    print("user 0 친구들 :", u['name'], u['id'] )
print("user 0 ", friends_of_friend_ids_bad(users[0]))
