import random
import win32com.client as wincl

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
    import sys
    import re
    try:
        dig = list(map(lambda i: int(re.findall(r'\d+', i)[0]), sys.argv[1:]))
        add_aloud(*dig)
    except IndexError:
        add_aloud(3,4)
            
