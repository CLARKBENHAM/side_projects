import random
import win32com.client as wincl
import sys
import re
import time

def math_q(a_range, b_range, tp = ("+", "-", "*", "/"), difficulty = "easy",
           negatives = False, num_decimal_digits = 0):
    "Baseline question generator"
    try:
        op = random.choice(tp)
    except:
        op = tp
    assert op in ("+", "-", "*", "/")

    def _rand(rng):
        v = random.randint(*rng)
        if num_decimal_digits > 0:
            v += random.randint(1, 10**num_decimal_digits - 1) / 10**num_decimal_digits
        if negatives and random.random() > 0.5:
                v *= -1
        return v

    if difficulty in ('medium', 'hard'):
        out = []
        for rng in (a_range, b_range):
            v = str(_rand(rng))
            while difficulty == 'medium' and (len(set(v)) <= len(v)//2
                                             or '0' in v
                                             or '1' in v):
                v = str(_rand(rng))
            while difficulty == 'hard' and (len(set(v)) <= len(v)//2
                                            or '0' in v
                                            or '1' in v
                                            or (op in ('*', '/')
                                                and v[-1] in ('0', '1', '5'))
                                            or (op in ('+', '-')
                                                and '0' in v)):
                v = str(_rand(rng))
            out += [float(v)]
        a,b = out
    else:
        a = _rand(a_range)
        b = _rand(b_range)
    if op == "-":
        a = a + b
    elif op == "/":
        a = a * b
    ans = eval(f"{a}{op}{b}")
    if difficulty == 'hard2':
        #assumes they're all ints!!
        sb = str(b)
        sa = str(a)
        num_digits_cycle = random.randint(1, min(len(str(a)), len(str(b))))
        if op == '-':
            # num_cycle = len(str(a)) > len(str(b))
            cycle_ix = [i for i in range(len(str(b))) if str(a)[-i] < str(b)[i]]
            if len(cycle_ix) < num_digits_cycle:
                new_ixs = random.sample(set(range(len(b))) - set(cycle_ix),
                                        num_digits_cycle - len(cycle_ix))
                for ix in new_ixs:
                    b += random.randint(int(str(a))[ix] - b + 1,
                                        10 - int(str(b)[ix]) -1) * 10**ix
        ans = eval(f"{a}{op}{b}")
    return a,b,op, ans

def hidden_math(argv, kwargs, num_qs = None, num_lines = 40):
    """Prints a bunch of lines after the question
    so have to keep all the numbers in your head.
    num_qs: Stops after this many q's
    num_lines: how far to print down after question; set to 0 for no printing
    """
    flash_time = 1.8
    num_right = 0
    num_guesses = 0

    def _finished(num_right, num_guesses, t = time.time()):
        t = time.time() - t
        print(f"Total Correct: {num_right}, in {t//60:.0f}' {t%60:.1f}\" ")
        print(f"Average Accuracy: {num_right/num_guesses *100:.1f}%.",
                  f"Average Time: {t/num_right:.2f}\" ")
        return None

    def _prnt(a,b,op):
        if a % 1 == 0:
            a = int(a)
        if b % 1 == 0:
            b = int(b)
        print(f"{a}  {op}  {b}")
        time.sleep(flash_time)
        print("\n"*num_lines)

    while True:
        a, b, op, ans = math_q(*argv, **kwargs)
        if random.random()>0.5:
            a,b = b,a
        _prnt(a,b,op)
        cnt = 1
        g = ans - 1
        while g != ans:
            if cnt %5 == 0:
                _prnt(a,b,op)
            g = input(": ")
            while 'r' in g:
                _prnt(a,b,op)
                g = input(": ")
            try:
                g = float(g)
            except:
                return _finished(num_right, num_guesses)
            num_guesses += 1
            cnt += 1
        print("Correct!\n\n")
        num_right +=1
        if num_qs and num_right == num_qs:
            return _finished(num_right, num_guesses)

# while True:
#     if False: #random.randint(1, 2) > 1:
#         math_q((1000, 10000), (1000, 10000), tp=("+", "-"))
#     else:

# hidden_math([(1003, 9999), (3, 9)], {'tp': ("*"),
#                                     'difficulty': 'hard',
#                                     'negatives': False,
#                                     'num_decimal_digits': 0})
hidden_math([(13, 99), (13, 99)], {'tp': ("*"), 'difficulty': 'hard'}, num_lines = 40)

# hidden_math([(13, 99), (13, 99)], {'tp': ("/"), 'difficulty': 'hard'}, num_qs = 3)
#%%
def kelly_bet(odds_range, num_qs = None, num_lines = 0,
              p_win_difficulty = "hard", p_win_num_decimal_digits = 0):
    """Gives Bet ratio and Odds and need to calculate kelly bet
    ods_range: list of tuples for each side of bet
    num_lines: Prints a bunch of lines after the question (make it disappear)
    num_qs: Stops after this many q's
    num_lines: how far to print down after question; set to 0 for no printing
    """
    flash_time = 1.8
    num_right = 0
    num_guesses = 0
    if isinstance(odds_range, tuple):
        odds_range = [(1,1), odds_range]
    kwargs = {'tp': ("+"),
                'difficulty': 'easy',
                'negatives': False,
                'num_decimal_digits': 0}

    def _finished(num_right, num_guesses, t = time.time()):
        t = time.time() - t
        print(f"Total Correct: {num_right}, in {t//60:.0f}' {t%60:.1f}\" ")
        print(f"Average Accuracy: {num_right/num_guesses *100:.1f}%.",
                  f"Average Time: {t/num_right:.2f}\" ")
        return None

    def _prnt(a,b, p_win):
        if a % 1 == 0:
            a = int(a)
        if b % 1 == 0:
            b = int(b)
        print(f"{a} : {b} with p_win = {p_win}")
        if num_lines != 0:
            time.sleep(flash_time)
            print("\n"*num_lines)

    while True:
        a, b, _, _ = math_q(*odds_range, **kwargs)
        p_win, _, _, _ =  math_q((4,10), (2,2), tp = ("+"),
                                 difficulty = p_win_difficulty,
                                 num_decimal_digits= p_win_num_decimal_digits)
        p_win /= 10
        _prnt(a, b, p_win)
        odds = b/a #normalized so your 1 : against
        ans = (p_win * (odds + 1) - 1) / odds
        ans = max(0, min(1, ans))#never > 1
        #rounding issue?
        cnt = 1
        g = 2
        while abs(g - ans) > 10**-9:
            if cnt %5 == 0:
                _prnt(a, b, p_win)
            g = input(": ")
            while 'r' in g:
                _prnt(a,b, p_win)
                g = input(": ")
            try:
                g = eval(g)
            except:
                return _finished(num_right, num_guesses)
            num_guesses += 1
            cnt += 1
        print("Correct!\n\n")
        num_right +=1
        if num_qs and num_right == num_qs:
            return _finished(num_right, num_guesses)
#p - q/b
kelly_bet([(3, 9), (9, 19)])
    #%%
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
            if cnt % 5 == 0:
                print(a,b)
        print(f"Correct!, {a} + {b} = {ans}\n")

#5x5 grid which flashes blue circles; have to click the cells which held blue
#click arrow which changed direction

#hit 1/0 if coloured circles on left or on right match

#hit 1/0 if center arrow of 5 points left vs. right (and line appears at different heights)

if __name__ == '__main__':
    try:
        digits = list(map(lambda i: int(re.findall(r'\d+', i)[0]), sys.argv[1:]))
        add_aloud(*digits)
    except IndexError:
        add_aloud(3,4)