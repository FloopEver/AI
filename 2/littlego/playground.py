# cd /Users/yijingyang/Documents/usc/2020spring/AI/hw/2/test
from host import GO

curr_state = "1111111111112111111111111"
state = []
for i in range(0, 25, 5):
    state.append([int(x) for x in curr_state[i: i+5]])

go = GO(5)
board = [
[1,1,1,1,1],
[2,2,2,2,2],
[0,0,0,0,0],
[0,0,0,0,0],
[0,0,0,0,0]
]
go.set_board(1, state, state)
print (state)
go.remove_died_pieces(2)
print (state)