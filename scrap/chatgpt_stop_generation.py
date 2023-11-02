# %% Use
# PyAutoGUI on mac has a flakey paste
# selenium doesn't work because Cloudflare blocks the connection
import time
import random
import string
import pyautogui as pygui
from pyautogui import Point
import subprocess
import os

os.chdir("/Users/clarkbenham/side_projects/scrap/")


# type in up to 4000 random characters
def paste_random():
    message = (
        "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(2000)) + "\n\r"
    )
    subprocess.run("pbcopy", text=True, input=message)
    pygui.hotkey("command", "v")
    time.sleep(0.5)
    pygui.hotkey("command", "v")  # 2nd paste works?!?
    print("pasted: ", message[:6])


# For a full-screened chrome chat wind on 16.1" mac pro
enter_msg_coord = Point(x=1001, y=1061)
generate_coord = Point(x=1378, y=998)  # same as stop coord
edit_coord = Point(x=1379, y=210)
save_and_submit_coord = Point(x=986, y=801)


# %%
def keep_editing_message_then_cancel(n_reps=60):
    """I got through a cycle of 60 before I hit timelimit.
    Canceling a message counts for less than submiting one
    """
    pygui.moveTo(enter_msg_coord, duration=0.3)
    pygui.click()
    paste_random()
    pygui.press("enter")
    time.sleep(0.4)
    pygui.moveTo(generate_coord)
    pygui.click()

    time.sleep(1)
    for i in range(n_reps):
        pygui.moveTo(edit_coord, duration=0.3)
        time.sleep(0.1)
        pygui.click(interval=0.1)
        pygui.move(-100, 0)
        pygui.click()

        pygui.hotkey("command", "a")
        time.sleep(0.5)
        pygui.hotkey("command", "a")
        paste_random()
        pygui.scroll(-1000)

        pygui.moveTo(save_and_submit_coord)
        pygui.click()

        time.sleep(0.5)
        pygui.moveTo(generate_coord)
        pygui.click()

        print("cycle", i)
        time.sleep(0.3)


def keep_starting_then_stopping(n_reps=60):
    """Keep clicking generate, then stop, then generate again.
    Got 30 retries in then was limited for 20 more minutes"""
    pygui.moveTo(edit_coord)  # selects window
    pygui.click(interval=0.1)

    for i in range(n_reps):
        pygui.moveTo(generate_coord)
        pygui.click()
        time.sleep(0.5)
        pygui.click()
        print("cycle", i)
        time.sleep(0.3)


# %%

if __name__ == "__main__":
    # input("Confirm you've manually sign in and started a new chat with GPT4...")
    # keep_editing_message_then_cancel()

    input("Confirm you've Typed a long message into GPT4 and hit Stop Generating...")
    keep_starting_then_stopping()
