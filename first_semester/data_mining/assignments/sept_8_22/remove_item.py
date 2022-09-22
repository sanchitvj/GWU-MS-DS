
def pop_item(list):
    list.pop(0)
    return list
    
if __name__ == "__main__":

    color = ["Red", "Black", "Green", "White", "Orange"]
    print("New list: ", pop_item(color))