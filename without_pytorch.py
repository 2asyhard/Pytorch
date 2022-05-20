'''
Solve XOR problem using

simple neural network without using deep learning library
only use random library to generate data
'''
import random


def multiply_matrix():
    '''
    3ways to multiply 2d list without numpy
    '''

    a = [[1,2],[3,4]]
    b = [[5,6],[7,8]]

    result1 = [[a[0][0]*b[0][0] + a[0][1]*b[1][0],
              a[0][0]*b[0][1] + a[0][1]*b[1][1]],
              [a[1][0]*b[0][0] + a[1][1]*b[1][0],
              a[1][0]*b[0][1] + a[1][1]*b[1][1]]]

    result2 = [[sum([a[0][k]*b[k][0] for k in range(2)]),
               sum([a[0][k]*b[k][1] for k in range(2)])],
               [sum([a[1][k]*b[k][0] for k in range(2)]),
               sum([a[1][k]*b[k][1] for k in range(2)])]]

    result3 = [[sum([a[i][k]*b[k][j] for k in range(2)]) for j in range(2)] for i in range(2)]

    print(result1)
    print(result2)
    print(result3)


def input_1_variable_SLP():
    '''
    SLP with single input variable / no hidden layers
    x -> y
    x*w = y
    '''

    def mul_matrix(mat1, mat2):
        result_mat = [[sum([mat1[i][k]*mat2[k][j] for k in range(len(mat2))]) for j in range(len(mat2[0]))] for i in range(len(mat1))]
        return result_mat

    def get_hypo(input_data, weight):
        x_data = [x[:1] for x in input_data]
        hypo = mul_matrix(x_data, weight)
        return hypo

    def get_new_weight(weight, lr, data):
        new_weight = weight[0][0] - lr*(1/len(data))*sum((weight[0][0]*data[i][0]-data[i][1])*data[i][0] for i in range(len(data)))
        return [[new_weight]]

    def get_loss(test_data, weight):
        hypo = get_hypo(test_data, weight)
        sum_loss = sum([(hypo[i][0]-test_data[i][-1])**2 for i in range(len(test_data))])
        return sum_loss/len(test_data)

    print('-')
    print('input1_variable_SLP')

    # generate test data [x, y]
    test_data = [[1,2],[2,4],[3,6],[4,8],[5,10],[6,12],[7,14],[8,16],[9,18],[10,20]]

    # initialize weights and parameters
    weight = [[random.random()*10]]
    learning_rate = 0.0001

    for training_iter in range(0,101):
        if training_iter % 10 == 0:
            # 현재 weight로 test 하기(loss값 구하기)
            loss = get_loss(test_data, weight)
            print(f'training iteration: {training_iter},  Loss: {loss}, Weight: {weight}')

        # generate random training data
        training_data = []
        for i in range(100):
            random_int = random.randint(1, i+10)
            training_data.append([random_int, random_int*2])

        # train weights
        weight = get_new_weight(weight, learning_rate, training_data)


def input_2_variables_SLP():
    '''
    2 input variable SLP
    x1, x2 -> y
    x1*w1 + x2*w2 = y
    '''
    def mul_matrix(mat1, mat2):
        result_mat = [[sum([mat1[i][k]*mat2[k][j] for k in range(len(mat2))]) for j in range(len(mat2[0]))] for i in range(len(mat1))]
        return result_mat

    def get_hypo(input_data, weight):
        x_data = [x[:2] for x in input_data]
        hypo = mul_matrix(x_data, weight)
        return hypo

    def get_new_weight(weight, lr, data_x, data_y):
        new_weight0 = weight[0][0] - lr*(1/len(data_x))*sum((weight[0][0]*data_x[i][0]+weight[1][0]*data_x[i][1]-data_y[i][0])*data_x[i][0] for i in range(len(data_x)))
        new_weight1 = weight[1][0] - lr*(1/len(data_x))*sum((weight[0][0]*data_x[i][0]+weight[1][0]*data_x[i][1]-data_y[i][0])*data_x[i][1] for i in range(len(data_x)))

        return [[new_weight0],[new_weight1]]

    def get_loss(test_data, weight):
        hypo = get_hypo(test_data, weight)
        sum_loss = sum([(hypo[i][0]-test_data[i][-1])**2 for i in range(len(test_data))])
        return sum_loss/len(test_data)


    def get_train_data():
        data_x = []
        data_y = []
        for i in range(100):
            random_int1 = random.randint(0, i + 10)
            random_int2 = random.randint(0, i + 20)
            data_x.append([random_int1, random_int2])
            data_y.append([(random_int1+random_int2)/2])
        return data_x, data_y


    print('-'*50)
    # generate test data [x1, x2, y]
    test_data = []
    for i in range(1, 11):
        data = [i, i*2, (i+i*2)/2]
        test_data.append(data)

    # weight and parameter settings
    weight = [[int(random.random()*10)] for _ in range(2)]
    learning_rate = 0.0005


    for training_iter in range(0,101):
        if training_iter % 10 == 0:
            # test, get loss
            loss = get_loss(test_data, weight)
            print(f'training iteration: {training_iter},  Loss: {loss}, Weight: {weight}')

        # generate training data [x, y]
        data_x, data_y = get_train_data()

        # weight 학습시키기
        weight = get_new_weight(weight, learning_rate, data_x, data_y)


def multi_variable_SLP():
    '''
    multi variable SLP with adjustable number of inputs
    x1, x2, ... xn -> y1, y2

    y1 = average of X
    y2 = sum of X
    '''

    def mul_matrix(mat1, mat2):
        result_mat = [[sum([mat1[i][k] * mat2[k][j] for k in range(len(mat2))]) for j in range(len(mat2[0]))] for i in
                      range(len(mat1))]
        return result_mat

    def get_hypo(x_data, weight):
        hypo = mul_matrix(x_data, weight)
        return hypo

    def get_new_weight(weight, lr, data_x, data_y):
        weights = [[weight[j][k] - lr * (1 / len(data_x)) * sum((sum([data_x[i][l] * weight[l][k] for l in range(len(data_x[0]))]) - data_y[i][k]) * data_x[i][j] for i in range(len(data_x))) for k in range(len(data_y[0]))] for j in range(len(data_x[0]))]
        return weights

    def get_loss(test_x, test_y, weight):
        hypo = get_hypo(test_x, weight)
        sum_loss = 0
        for i in range(len(hypo)):
            for j in range(len(hypo[0])):
                sum_loss += (hypo[i][j] - test_y[i][j]) ** 2

        return sum_loss / (len(hypo) * len(hypo[0]))


    def get_data(input_num, data_size):
        test_x = []
        test_y = []
        for i in range(data_size):
            tmp_list = []
            for j in range(1, input_num + 1):
                tmp_val = random.randint(j, j * 2 + 2) * random.randint(j, j * 2 + 2)
                tmp_list.append(tmp_val)
            test_x.append(tmp_list)
            test_y.append([])
            test_y[-1].append(sum(tmp_list) / len(tmp_list))
            test_y[-1].append(sum(tmp_list))

        return test_x, test_y


    # initialize weight, parameters
    input_num = 4 # [x1, x2, ... xn]
    output_num = 2 # should be fixed, [average of X, sum of X]
    weight = [[int(random.random() * 10) for _ in range(output_num)] for _ in range(input_num)]
    learning_rate = 0.0005

    # generate test data
    test_x, test_y = get_data(input_num, 10)

    for training_iter in range(0, 1001):
        # check test results
        if training_iter % 100 == 0:
            loss = get_loss(test_x, test_y, weight)
            print(f'training iteration: {training_iter},  Loss: {loss},\n Weight: {weight}')
            print('-' * 40)

        # generate training data
        data_x, data_y = get_data(input_num, 100)

        # train weights
        weight = get_new_weight(weight, learning_rate, data_x, data_y)


def single_var_input_1h_MLP():
    '''
    MLP with 1 single variable input/1 hidden layer/no activation
    x -> h1 -> y(=2*x)
    x*w1 = h1
    h1*w2 = y
    use chain rule
    '''
    def mul_matrix(mat1, mat2):
        result_mat = [[sum([mat1[i][k] * mat2[k][j] for k in range(len(mat2))]) for j in range(len(mat2[0]))] for i in
                      range(len(mat1))]
        return result_mat

    def get_hypo(x_data, weight):
        hypo = mul_matrix(x_data, weight)
        return hypo

    def get_new_weight(weights, lr, data_x, data_y):
        new_weight1 = [weights[0][0][0] - lr*(1/len(data_x)) * sum([data_x[i][0]*weights[1][0][0] *2*(data_x[i][0]*weights[0][0][0]*weights[1][0][0] - data_y[i][0]) for i in range(len(data_x))])]
        new_weight2 = [weights[1][0][0] - lr*(1/len(data_x)) * sum([data_x[i][0]*weights[0][0][0] *2*(data_x[i][0]*weights[0][0][0]*weights[1][0][0] - data_y[i][0]) for i in range(len(data_x))])]
        new_weights = [[new_weight1], [new_weight2]]

        return new_weights

    def get_loss(test_x, test_y, weights):
        input_g1 = test_x
        for weight in weights:
            output_g2 = get_hypo(input_g1, weight)
            input_g1 = output_g2

        sum_loss = 0
        for i in range(len(output_g2)):
            for j in range(len(output_g2[0])):
                sum_loss += (output_g2[i][j] - test_y[i][j]) ** 2

        # sum_loss = sum([(hypo[i][0]-test_data[i][-1])**2 for i in range(len(test_data))])
        return sum_loss / (len(output_g2) * len(output_g2[0]))


    def get_data(input_num, data_size):
        data_x = []
        data_y = []
        for i in range(data_size):
            tmp_list = []
            for j in range(1, input_num + 1):
                tmp_val = random.randint(j, j * 2 + 2) * random.randint(j, j * 2 + 2)
                tmp_list.append(tmp_val)
            data_x.append(tmp_list)
            data_y.append([])
            data_y[-1].append(data_x[-1][0]*2)

        return data_x, data_y


    # fixed parameters
    input_num = 1
    hidden_layer1_num = 1
    output_num = 1

    # get test data
    test_x, test_y = get_data(input_num, 10)

    # weight settings
    weight1 = [[int(random.random() * 10) for _ in range(hidden_layer1_num)] for _ in range(input_num)]
    weight2 = [[int(random.random() * 10) for _ in range(output_num)] for _ in range(hidden_layer1_num)]
    weights = [weight1, weight2]

    # learning rate setting
    learning_rate = 0.0001

    for training_iter in range(0, 101):
        if training_iter % 10 == 0:
            # 현재 weight로 test 하기(loss값 구하기)
            loss = get_loss(test_x, test_y, weights)
            print(f'training iteration: {training_iter},  Loss: {loss},\n Weight: {weights}')
            print('-' * 40)

        # 학습 데이터 (0, 100) 범위의 값 랜덤 생성 [x, y]
        data_x, data_y = get_data(input_num, 100)

        # weight 학습시키기
        weights = get_new_weight(weights, learning_rate, data_x, data_y)

single_var_input_1h_MLP()