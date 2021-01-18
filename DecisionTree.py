# The second assignment of the Introduction to Pattern Recognition
# Generate a Decision Tree from the data based on information entropy
# The data is given in table 4.3 of our textbook "Machine Learning", on page 84
# This project ignore the feature "Sugar Content" in table 4.3


from math import log
from copy import deepcopy


def load_data():
        """
        Load the data first
        This function load the data used for training the Decision Tree
        """
        # All features considered to train the decision tree
        # features = ['number', 'color', 'root', 'knock_sound', 'texture', 'belly_button', 'touch', 'density']
        # The last one of each data indicates whether it's a good watermelon or not
        # Label "good" and "bad"
        # The data of each sample is a dictionary, and each dictionary is an element of the list "data"
        data = [{'number': 1, 'color': 'green', 'root': 'curled', 'knock_sound': 'turbidity', 'texture': 'clear', 'belly_button': 'hollow', 'touch': 'hard', 'density': 0.697, 'label': 'good'},
                {'number': 2, 'color': 'black', 'root': 'curled', 'knock_sound': 'dull', 'texture': 'clear', 'belly_button': 'hollow', 'touch': 'hard', 'density': 0.774, 'label': 'good'},
                {'number': 3, 'color': 'black', 'root': 'curled', 'knock_sound': 'turbidity', 'texture': 'clear', 'belly_button': 'hollow', 'touch': 'hard', 'density': 0.634, 'label': 'good'},
                {'number': 4, 'color': 'green', 'root': 'curled', 'knock_sound': 'dull', 'texture': 'clear', 'belly_button': 'hollow', 'touch': 'hard', 'density': 0.608, 'label': 'good'},
                {'number': 5, 'color': 'white', 'root': 'curled', 'knock_sound': 'turbidity', 'texture': 'clear', 'belly_button': 'hollow', 'touch': 'hard', 'density': 0.556, 'label': 'good'},
                {'number': 6, 'color': 'green', 'root': 'little_curled', 'knock_sound': 'turbidity', 'texture': 'clear', 'belly_button': 'little_hollow', 'touch': 'soft', 'density': 0.403, 'label': 'good'},
                {'number': 7, 'color': 'black', 'root': 'little_curled', 'knock_sound': 'turbidity', 'texture': 'little_blurry', 'belly_button': 'little_hollow', 'touch': 'soft', 'density': 0.481, 'label': 'good'},
                {'number': 8, 'color': 'black', 'root': 'little_curled', 'knock_sound': 'turbidity', 'texture': 'clear', 'belly_button': 'little_hollow', 'touch': 'hard', 'density': 0.437, 'label': 'good'},
                {'number': 9, 'color': 'black', 'root': 'little_curled', 'knock_sound': 'dull', 'texture': 'little_blurry', 'belly_button': 'little_hollow', 'touch': 'hard', 'density': 0.666, 'label': 'bad'},
                {'number': 10, 'color': 'green', 'root': 'straight', 'knock_sound': 'crisp', 'texture': 'clear', 'belly_button': 'flat', 'touch': 'soft', 'density': 0.243, 'label': 'bad'},
                {'number': 11, 'color': 'white', 'root': 'straight', 'knock_sound': 'crisp', 'texture': 'blurry', 'belly_button': 'flat', 'touch': 'hard', 'density': 0.245, 'label': 'bad'},
                {'number': 12, 'color': 'white', 'root': 'curled', 'knock_sound': 'turbidity', 'texture': 'blurry', 'belly_button': 'flat', 'touch': 'soft', 'density': 0.343, 'label': 'bad'},
                {'number': 13, 'color': 'green', 'root': 'little_curled', 'knock_sound': 'turbidity', 'texture': 'little_blurry', 'belly_button': 'hollow', 'touch': 'hard', 'density': 0.639, 'label': 'bad'},
                {'number': 14, 'color': 'white', 'root': 'little_curled', 'knock_sound': 'dull', 'texture': 'little_blurry', 'belly_button': 'hollow', 'touch': 'hard', 'density': 0.657, 'label': 'bad'},
                {'number': 15, 'color': 'black', 'root': 'little_curled', 'knock_sound': 'turbidity', 'texture': 'clear', 'belly_button': 'little_hollow', 'touch': 'soft', 'density': 0.360, 'label': 'bad'},
                {'number': 16, 'color': 'white', 'root': 'curled', 'knock_sound': 'turbidity', 'texture': 'blurry', 'belly_button': 'flat', 'touch': 'hard', 'density': 0.593, 'label': 'bad'},
                {'number': 17, 'color': 'green', 'root': 'curled', 'knock_sound': 'dull', 'texture': 'little_blurry', 'belly_button': 'little_hollow', 'touch': 'hard', 'density': 0.719, 'label': 'bad'}]
        # All possible values for each feature
        # feature_values = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        #                   ['green', 'black', 'white'],
        #                   ['curled', 'little_curled', 'straight'],
        #                   ['turbidity', 'dull', 'crisp'],
        #                   ['clear', 'little_blurry', 'blurry'],
        #                   ['hollow', 'little_hollow', 'flat'],
        #                   ['hard', 'soft'],
        #                   [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719]]
        return data


def calculate_information_entropy(current_data):
        """
        The input is full data of each sample in current step
        The output is the information entropy of this sample set
        """
        total_number_of_samples = len(current_data)
        # Create an empty dictionary to record all labels and their appearance number in current_data
        number_of_each_label = {}
        for sample in current_data:
                # label is the last element of a sample data
                label = sample['label']
                if label not in number_of_each_label.keys():
                        # Create a new label type
                        number_of_each_label[label] = 0
                # count the label number
                number_of_each_label[label] = number_of_each_label[label] + 1
        information_entropy = 0.0
        for label in number_of_each_label.keys():
                pk = float(number_of_each_label[label]/total_number_of_samples)
                information_entropy = information_entropy - pk * log(pk, 2)
        return information_entropy


def find_dividing_point_and_calculate_IG(current_data, feature):
        """
        The input current_data is the data of each sample in current step
        The input feature is a specific feature of current_data
        Here, "feature" should be a continuous feature, discrete feature does not need a dividing point
        The function return the best dividing point and its corresponding Information Gain (IG)
        """
        # Get all values of "feature" shown in all samples
        all_value = []
        for sample in current_data:
                all_value.append(sample[feature])
        # make "all_value" from small to big
        all_value.sort()
        # Get the length of "all_value", as well as the total number of samples in current_data
        length = len(all_value)
        # Calculate the potential dividing point
        potential_dividing_point = []
        for i in range(length-1):
                # i = 0,1,2,...,length-2
                temp = 1/2*(all_value[i] + all_value[i+1])
                potential_dividing_point.append(temp)
        # Initialize the best dividing point and max information gain (IG)
        best_dividing_point = -1
        max_gain = -1
        # For all potential dividing points t, calculate Gain(D,a,t), update best_dividing_point and max_gain
        for t in potential_dividing_point:
                # Initialize Gain(D,a) as Ent(D)
                Gain = calculate_information_entropy(current_data)
                # Initialize sub_data_1 and sub_data_2, which are divided by current dividing point t
                sub_data_1 = []
                sub_data_2 = []
                for sample in current_data:
                        if sample[feature] <= t:
                                sub_data_1.append(sample)
                        else:
                                sub_data_2.append(sample)
                length_of_sub_data_1 = len(sub_data_1)
                length_of_sub_data_2 = len(sub_data_2)
                Ent_of_sub_data_1 = calculate_information_entropy(sub_data_1)
                Ent_of_sub_data_2 = calculate_information_entropy(sub_data_2)
                Gain = Gain - length_of_sub_data_1/length*Ent_of_sub_data_1 - length_of_sub_data_2/length*Ent_of_sub_data_2
                if Gain > max_gain:
                        best_dividing_point = t
                        max_gain = Gain
        return best_dividing_point, max_gain


def split_data_for_discrete_feature(current_data, feature, value):
        """
        The input current_data is the data of each sample in current step
        The input feature is a specific feature of current_data, which should be discrete
        The input value is a specific value of the specific feature
        This function split current_data, return the sub_data including samples with given value in given feature
        """
        # initialize the data after split with an empty list
        data_after_split = []
        for sample in current_data:
                if sample[feature] == value:
                        # These samples meet the requirement
                        # Notice that, deepcopy should be used here, or there will be error after deleting one of the feature in build_a_decision_tree function
                        # Simply use append will lead to the error
                        deepcopy_sample = deepcopy(sample)
                        data_after_split.append(deepcopy_sample)
        return data_after_split


def split_data_for_continuous_feature(current_data, feature, dividing_point):
        """
        The input current_data is the data of each sample in current step
        The input feature is a specific feature of current_data, which should be continuous, for example, density
        The input dividing_point is the basis for splitting, which should be calculated by function find_dividing_point_and_calculate_IG
        The outputs are sub_data_1 and sub_data_2
        Samples in sub_data_1 have "value" <= dividing_point in "feature", while samples in sub_data_2 have "value" > dividing_point
        """
        # Initialize sub_data_1 and sub_data_2
        sub_data_1 = []
        sub_data_2 = []
        for sample in current_data:
                value = sample[feature]
                # Deepcopy is needed, like in function split_data_for_discrete_feature, or may lead to the error
                if value <= dividing_point:
                        deepcopy_sample = deepcopy(sample)
                        sub_data_1.append(deepcopy_sample)
                else:
                        deepcopy_sample = deepcopy(sample)
                        sub_data_2.append(deepcopy_sample)
        return sub_data_1, sub_data_2


def calculate_information_gain(current_data, chosen_feature):
        """
        The input current_data is the data of each sample in current step
        The input chosen_feature is the feature chosen to calculate information gain (IG)
        The output is the IG of chosen_feature in current_data
        The bigger IG is, the better chosen_feature is to be the split feature
        There are two kinds of features, discrete or continuous, they need to calculate differently
        """
        # get the total number of samples in current_data (|D|)
        total_number_of_samples = len(current_data)
        # Calculate the total information entropy of "current_data" (Ent(D))
        EntD = calculate_information_entropy(current_data)
        # Create an empty dictionary to record all values of chosen_feature appearance number in current_data
        # That is, to record |Dv| for each v (value)
        number_of_chosen_feature_value = {}
        if chosen_feature not in ['density']:
                # This means chosen_feature is discrete
                # Initialize the information gain Gain(D,a)
                Gain = EntD
                for sample in current_data:
                        value = sample[chosen_feature]
                        if value not in number_of_chosen_feature_value.keys():
                                # Create a new value type
                                number_of_chosen_feature_value[value] = 0
                        number_of_chosen_feature_value[value] = number_of_chosen_feature_value[value] + 1
                for value in number_of_chosen_feature_value.keys():
                        sub_data = split_data_for_discrete_feature(current_data, chosen_feature, value)
                        Dv_D = float(number_of_chosen_feature_value[value]/total_number_of_samples)
                        # Calculate the information entropy of samples in Dv (Ent(Dv))
                        Ent_Dv = calculate_information_entropy(sub_data)
                        Gain = Gain - Dv_D*Ent_Dv
                return Gain
        else:
                # This means chosen_feature is continuous
                dividing_point, Gain = find_dividing_point_and_calculate_IG(current_data, chosen_feature)
                return Gain


def choose_best_split_feature(current_data):
        """
        The input current_data is the data of each sample in current step
        This function return the best split feature, that is, with the biggest Information Gain
        """
        # First, get all potential split features
        the_first_sample = current_data[0]
        potential_split_features = the_first_sample.keys()
        # Pay attention, 'label' and 'number' is also in potential_split_features
        # Initialize the best split feature and the max information gain
        best_split_feature = 'none'
        max_gain = -1
        # Try all potential features, update best_split_feature and max_gain
        for feature in potential_split_features:
                if feature not in ['label', 'number']:
                        # 'label' and 'number' actually are not a feature
                        Gain = calculate_information_gain(current_data, feature)
                        # print(feature + '  ', Gain)  # This line is for testing
                        if Gain > max_gain:
                                max_gain = Gain
                                best_split_feature = feature
        return best_split_feature


def Vote(current_data):
        """
        The input current_data is the data of each sample in current step
        This function return the label appearance the most in "current_data"
        """
        # Create an empty list to store all labels of samples in current_data
        labels = []
        for sample in current_data:
                labels.append(sample['label'])
        # Record the numbers of each kind of label
        number_of_each_label = {}
        for label in labels:
                if label not in number_of_each_label.keys():
                        number_of_each_label[label] = 0
                number_of_each_label[label] = number_of_each_label[label] + 1
        # Initialize the most appeared label and the most appeared time
        most_appeared_label = 'none'
        most_appeared_time = 0
        for label in number_of_each_label.keys():
                if number_of_each_label[label] > most_appeared_time:
                        most_appeared_time = number_of_each_label[label]
                        most_appeared_label = label
        return most_appeared_label


def build_a_decision_tree(initial_data):
        """
        This function build a decision tree end-to-end
        The input "initial_data" is all complete training data, and the output is the decision tree trained based on the data
        The result (the decision tree) is stored in a dictionary
        """
        # Get the total number of samples
        total_number_of_samples = len(initial_data)
        # list label of all samples one by one
        label_list = [sample['label'] for sample in initial_data]
        if label_list.count(label_list[0]) == total_number_of_samples:
                # All samples have the same label, so no more branch
                return label_list[0]

        # Get the first sample to see what features we still have
        the_first_sample = initial_data[0]
        all_features = list(the_first_sample.keys())
        # print(all_features)  # for testing
        all_features.remove('label')
        all_features.remove('number')
        if len(all_features) == 0:
                # No more features for branching, use the most appeared label as the decision result
                return Vote(initial_data)

        # If neither of circumstance above is met, we need to find a best feature and branch the tree
        best_split_feature = choose_best_split_feature(initial_data)
        # print(best_split_feature)  # for testing
        my_tree = {best_split_feature: {}}
        # print(my_tree)  # for testing
        if best_split_feature not in ['density']:
                # This means best_split_feature is discrete
                # print('discrete feature')  # for testing
                # Get all values of best_split_feature shown in "initial_data"
                possible_value = []
                for sample in initial_data:
                        if sample[best_split_feature] not in possible_value:
                                possible_value.append(sample[best_split_feature])
                # Split initial_sample and branch
                # print(possible_value)  # for testing
                for value in possible_value:
                        sub_data = split_data_for_discrete_feature(initial_data, best_split_feature, value)
                        for each_sample in sub_data:
                                # We need to delete best_split_feature in sub_data, because the best_split_feature is discrete
                                del each_sample[best_split_feature]
                        my_tree[best_split_feature][value] = build_a_decision_tree(sub_data)
                return my_tree
        else:
                # This means best_split_feature is continuous
                best_dividing_point, max_gain_useless_here = find_dividing_point_and_calculate_IG(initial_data, best_split_feature)
                sub_data_1, sub_data_2 = split_data_for_continuous_feature(initial_data, best_split_feature, best_dividing_point)
                # We do not need to delete best_split_feature in sub_data, because the best_split_feature is continuous
                possible_value = [best_split_feature + '<=' + str(best_dividing_point), best_split_feature + '>' + str(best_dividing_point)]
                for i in range(2):
                        # i = 0, 1
                        if i == 0:
                                value = possible_value[0]
                                my_tree[best_split_feature][value] = build_a_decision_tree(sub_data_1)
                        else:
                                value = possible_value[1]
                                my_tree[best_split_feature][value] = build_a_decision_tree(sub_data_2)
                        # print(my_tree)  # for testing
                return my_tree


if __name__ == '__main__':
        data = load_data()
        my_tree = build_a_decision_tree(data)
        print(my_tree)


# The main function below is used to evaluate the decision tree algorithm
# if __name__ == '__main__':
#         data = load_data()
#         # Initialize the test_data and eval_data
#         test_data = []
#         eval_data = []
#         for sample in data:
#                 if sample['number'] in [1, 2, 3, 6, 7, 10, 14, 15, 16, 17]:
#                         test_data.append(sample)
#                 else:
#                         eval_data.append(sample)
#         my_tree = build_a_decision_tree(test_data)
#         print(my_tree)