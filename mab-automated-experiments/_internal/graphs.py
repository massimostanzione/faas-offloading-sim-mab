import matplotlib.pyplot as plt

def line_graph(data:dict, title:str=""):

    for item in data.items():
        key=item[0]
        value=item[1]
        print (item)
        print ("key", key)
        print ("value", value)
        plt.plot(value["x"], value["y"], label=key)
    plt.legend()
    #plt.plot(value["x"], value["y"], label=key)
    #plt.plot(datadict.get("x"), datadict.get("y"), label=datadict.)
    plt.xlabel("fattore di esplorazione")  # add X-axis label
    plt.ylabel("(var.) tra le scelte dei bracci")  # add Y-axis label
    plt.title(title)  # add title
    return plt