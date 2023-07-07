def assign_meal(row):
    if row['home(meal)'] == 1:
        return 1
    else :
        return 0
def assign_other(row):
    if row['home(other)'] == 1:
        return 1
    else :
        return 0
def assign_out(row):
    if row['out'] == 1:
        return 1
    else :
        return 0
def assign_sleep(row):
    if row['home(sleep)'] == 1:
        return 1
    else :
        return 0
def assign_all(row):
    if row['home(meal)'] == 1:
        return 0
    if row['home(other)'] == 1:
        return 1
    if row['out'] == 1:
        return 2
    if row['home(sleep)'] == 1:
        return 3
    return 3

assign={'sleep':assign_sleep,'out':assign_out,'meal':assign_meal,'other':assign_other,'all':assign_all}
