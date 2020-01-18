import json
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from HW1.smu.qlearn import Catch

if __name__ == "__main__":
    # Make sure this grid size matches the value used fro training
    grid_size = 10

    with open("/Users/pankaj/dev/git/smu/ML2/HW1/smu/model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("/Users/pankaj/dev/git/smu/ML2/HW1/smu/model.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    env = Catch(grid_size)
    c = 0
    plt.show()
    for e in range(10):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        plt.imshow(input_t.reshape((grid_size,)*2),
                   interpolation='none', cmap='gray')
        plt.savefig("%03d.png" % c)
        c += 1
        while not game_over:
            input_tm1 = input_t

            # get next action
            q = model.predict(input_tm1)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)

            plt.imshow(input_t.reshape((grid_size,)*2),
                       interpolation='none', cmap='gray')
            plt.show()
            plt.savefig("%03d.png" % c)
            c += 1
