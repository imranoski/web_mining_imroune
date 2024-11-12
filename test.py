def text_to_list(string):
    string += " "
    liste = []
    word = ""
    for i in range(len(string)):
        if string[i] != " ":
            word += string[i]
        else :
            liste.append(word)
            word = ""   
    return liste

phrase = "J'aime beaucoup mes amis"
phrase2 = "Mais parfois c'est vraiment pas des gens biens"
print(text_to_list(phrase2))
        



