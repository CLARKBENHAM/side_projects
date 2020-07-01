import random
import win32com.client as wincl
import sys
import re
import time

def add_aloud(numa, numb):
    "Given the number of digits in the first <= number of digits of second digit to be based asks addition on them"
    speak = wincl.Dispatch("SAPI.SpVoice")
    print(numa, numb)
    while True:
        a = random.randint(3*(10**numa),8*(10**numa))
        b = random.randint(8*(10**(numb-1)), 88*(10**(numb-1)))
        ans = a+b
        g = 0
        if a%7 <= 3:
            b,a = a,b
        speak.Speak(f'{a}, + {b}')
        cnt = 0
        while g!= ans:
            g = input(": ")
            if 'r' in g:
                speak.speak(f'{a}, + {b}')
            else:
                g = int(g)
            cnt += 1
            if cnt > 5:
                print(a,b)
        print(f"Correct!, {a} + {b} = {ans}\n")

if __name__ == '__main__':
    try:
        digits = list(map(lambda i: int(re.findall(r'\d+', i)[0]), sys.argv[1:]))
        add_aloud(*digits)
    except IndexError:
        add_aloud(3,4)

#%%
def math_q(a_range, b_range, tp=("+", "-", "*", "/"), flash_time=1):
    "Baseline question generator"
    try:
        op = random.choice(tp)
    except:
        op = tp
    assert op in ("+", "-", "*", "/")
    a = random.randint(*a_range)
    b = random.randint(*b_range)
    if random.randint(0, 2)>1:
        a,b = b,a
    if op == "-":
        a = a + b
    elif op == "/":
        a = a * b
    print(f"{a}  {op}  {b}")
    ans = eval(f"{a}{op}{b}")
    g = ans - 1
    time.sleep(flash_time)
    # input("proced?")
    print("\n"*40)
    cnt = 1
    while g != ans:
        g = input(": ")
        if 'r' in g or cnt % 5 == 0:
            print(f"{a}  {op}  {b}")
            time.sleep(flash_time)
            print("\n"*40)
        else:
            g = int(g)
        cnt += 1
    print(f"Correct!, {a} {op} {b} = {ans}\n")

while True:
    if False: #random.randint(1, 2) > 1:
        math_q((1000, 10000), (1000, 10000), tp=("+", "-"))
    else:
        math_q((13, 40), (11, 40), tp=("*", "/"))
    #%%
