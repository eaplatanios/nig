import time

from nig.utilities.generic import elapsed_timer

def test_elapsed_timer():
    sleep_time = 0.01 # sec
    with elapsed_timer() as elapsed:
        time.sleep(sleep_time)
        t_middle_1 = elapsed()
        time.sleep(sleep_time)
        t_middle_2 = elapsed()
        time.sleep(sleep_time)
    t_final_1 = elapsed()
    time.sleep(sleep_time)
    t_final_2 = elapsed()

    assert t_middle_1 < t_middle_2 < t_final_1
    assert t_final_1 == t_final_2
