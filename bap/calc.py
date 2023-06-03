import time
start_time = time.time()

it_time = 923/10

print("ex1.1 steps, N=50:", it_time*50*8/60/60)
print("ex1.2.1 steps=5, N=20:", it_time*20*3/60+3*(1+2+3+5+10)+5)

# even if optimal step and particle number are different, use these parameters to not waste time
#the parameters don't impact the optimal m and simulation start time
# simulation number and start time per computer
print("ex1.2.1 steps=3, N=5:", (it_time*5*3/60+3*(1+2+3+5+10)+5)*5/60)

# optimal start sim time and simulation number
print("ex1.2.2 steps=3, N=5, simnumber=5, m={0.1, 0.5, 1, 2, 5, 10, 25}:", (it_time*5*3/60+10*5+5)*7*5/60)

# optimal step + particle number, simulation number and start time, don't do it 5 times so per computer
print("ex2.1.1 steps=3, N=10 number of agents={1,2,5,10,25}:", (it_time*10*3/60+10*(2+3+4+5+10+25)+5)/60)
#don'tdo it 5 times so per computer
print("ex2.1.2 steps=3, N=10 number of agents={1,2,3,4,5}:", (it_time*10*3/60+10)*(2+3+4+5)/60/4)

#volume experiment, optimal step + particle number, simulation number and start time, don't do it 5 times so per computer
print("ex2.2 steps=3, N=10, number of agents={1,5}, volume={?, 10 values}:", (it_time*10*3/60*2+10*11+5)/60)




end_time = time.time()

