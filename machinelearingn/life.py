from string import ascii_lowercase
import random,itertools, keyboard


print("Enter a phrase with only letters.")
phrase = input("Enter a phrase: ").lower().strip(" ").strip("\n")

word_p = ""
species_list = []

lengphrase = len(phrase)

for letter in phrase:
    if letter != " ":
        word_p += letter
    else:
        species_list.append(word_p)
        word_p = ""

if word_p not in species_list:
    species_list.append(word_p)

print(species_list)

guess_dic = {}
final_guess_dic = {}
tries = 0
tries_list = []
answer = []


def generate_specie(guess_dic, lengword):

    guess = ""
    
    for i in range(18):
        for i in range(lengword):
            guess += random.choice(ascii_lowercase)

        guess_dic[guess] = ""
        guess = ""

    return guess_dic


def choose_evolution(guess_dic, lengword, word):

    percent = 0
    lenguess_dic = len(guess_dic)

    for i,fguess in enumerate(guess_dic):
    
            for index, letter in enumerate(list(guess_dic)[i]):
    
                if letter == word[index]:
                    percent += 100 / lengword
    
            guess_dic[fguess] = percent
            percent = 0

    guess_dic = dict(sorted(guess_dic.items(), key=lambda item: item[1], reverse=True))

    for animal in range(lenguess_dic):
        if  len(guess_dic) > 3:
            guess_dic.popitem()

    items = list(guess_dic.items())
    random.shuffle(items)

    guess_dic = dict(items)

    return guess_dic


def breed_animal(guess_dic, lengword):

    poss_l = []
    poss_l_ori = []

    for animal in guess_dic:

        lenanimal = len(animal)
        breeding_brute = []

        part1 = animal[:lenanimal//2]
        part2 = animal[lenanimal//2:]
        poss_l.append(part1)
        poss_l.append(part2)
        poss_l_ori.append(part1)
        poss_l_ori.append(part2)

    for lenght in range(1, len(poss_l)+1):
        for combination in itertools.permutations(poss_l, lenght):
            result = "".join(combination)

            if len(result) in (lengword-1, lengword) and result not in poss_l_ori:
                breeding_brute.append(result)
            
    return breeding_brute


def mutation(breeding_brute, lengword):

    for index,animal in enumerate(breeding_brute):
    
            random_mut = random.randint(0, 100)
            random_char = random.randint(0, len(animal)-1)
    
            if random_mut < 50:
                breeding_brute[index] = breeding_brute[index].replace(animal[random_char], random.choice(ascii_lowercase), 1)

    for index,animal in enumerate(breeding_brute):
        if len(animal) == lengword-1:
            breeding_brute[index] += random.choice(ascii_lowercase)


def check_population(breeding_brute, guess_dic):

    for animal in breeding_brute:
        guess_dic[animal] = ""

    return guess_dic

for word in species_list:

    lengword = len(word)
    a = 0
    guess_dic = {}
    final_guess_dic = {}
    tries = 0

    print(word)

    while word not in final_guess_dic:

        if a == 0:
            generate_specie(guess_dic, lengword)
            a += 1

        guess_dic = choose_evolution(guess_dic, lengword, word)
        print("\n", guess_dic)
        breeding_brute = breed_animal(guess_dic, lengword)
        mutation(breeding_brute, lengword)
        final_guess_dic = check_population(breeding_brute, guess_dic)
        print("\n", final_guess_dic)

        tries += 1

        if keyboard.is_pressed('f'):
            break
        else:
            continue

    for index, animal in enumerate(final_guess_dic):
            if animal == word:
                tries_list.append(tries)
                answer.append(animal)
                break

print("\n\n")

for index, animal in enumerate(answer):
    print(f"The word {animal} was guessed in {tries_list[index]} tries.\n")

print(f"The Phrase is {phrase}.")