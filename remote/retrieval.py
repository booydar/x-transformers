import numpy as np

# K = 4

def get_three_letters(K):
    replace = K > 26
    return np.random.choice(range(11,26+11), K, replace=replace)

<<<<<<< HEAD
def get_three_numbers(K):
    replace = K > 10
    return np.random.choice(range(10), K, replace=replace)

def create_sequence(K, one_hot=True):
    letters = get_three_letters(K)
    numbers = get_three_numbers(K)
    X = np.zeros((2*K + 1))
=======
def create_sequence(one_hot=True):
    letters = get_three_letters()
    numbers = get_three_numbers()
    X = np.zeros((9-2))
>>>>>>> 7ba9a1fb194a7c76521d60a5bd3409c5ea94d9f4
    y = np.zeros((1))
    for i in range(0, 2*K, 2):
        X[i] = letters[i//2]
        X[i+1] = numbers[i//2]

    # append ??
<<<<<<< HEAD
#     X[6] = 10
#     X[7] = 10

    # last key and respective value (y)
    index = np.random.choice(range(0,K), 1, replace=False)
    X[2*K] = letters[index]
=======
    # X[6] = 10
    # X[7] = 10

    # last key and respective value (y)
    index = np.random.choice(range(0,3), 1, replace=False)
    X[8-2] = letters[index]
>>>>>>> 7ba9a1fb194a7c76521d60a5bd3409c5ea94d9f4
    y = numbers[index]

    if one_hot:
        # one hot encode X and y
        X_one_hot = np.eye(26+10+1)[np.array(X).astype('int')]
        y_one_hot = np.eye(26+10+1)[y][0]

        return X_one_hot, y_one_hot
    else:
        return X, y

def ordinal_to_alpha(sequence):
    """
    Convert from ordinal to alpha-numeric representations.
    Just for funsies :)
    """
    corpus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
              'a','b','c','d','e','f','g','h','i','j','k','l',
              'm','n','o','p','q','r','s','t','u','v','w','x','y','z','?']

    conversion = ""
    for item in sequence:
        conversion += str(corpus[item.argmax()])
    return conversion