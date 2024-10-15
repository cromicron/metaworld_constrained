# The metaworld benchmark only has a train and a testset. This is a problem, as
# Success Metrics are evaluated by the best performing result on the test set.
# However, this leads to the problem, that we don't know how well the agent
# would perform on new tasks. For ML10 we can choose validation tasks from the
# remaining unused 35 tasks. We chose 6 tasks - 3 where no picking objects is
# necessary and 3 where objects must be picked and moved at random

import random
random.seed(789256)
envs_pick = [
    "stick-pull-v2",
    "bin-picking-v2",
    "coffee-push-v2",
    "plate-slide-side-v2",
    "disassemble-v2",
    "box-close-v2",
    "soccer-v2",
    "hand-insert-v2",
    "stick-push-v2",
    "coffee-pull-v2",
    "assembly-v2",
    "hammer-v2",
    "pick-out-of-hole-v2",
    "push-wall-v2",
    "push-back-v2",
    "plate-slide-v2",

]

ambivalent = [
    "handle-pull-side-v2",
    "peg-unplug-side-v2",
    "plate-slide-back-side-v2",
    "plate-slide-back-v2",
]
envs_non_pick = [
    "handle-press-v2",
    "dial-turn-v2",
    "button-press-topdown-wall-v2",
    "faucet-open-v2",
    "handle-pull-v2",
    "reach-wall-v2",
    "door-unlock-v2",
    "window-close-v2",
    "handle-press-side-v2",
    "pick-place-wall-v2",
    "door-lock-v2",
    "button-press-v2",
    "faucet-close-v2",
    "coffee-button-v2",
    "button-press-wall-v2"
]
envs_valid_pick = random.sample(envs_pick, 3)
envs_valid_non_pick = random.sample(envs_non_pick, 3)
envs_valid = envs_valid_pick + envs_valid_non_pick
envs_valid = random.sample(envs_valid, len(envs_valid))
# handle_press turned out to be subomtimal, as there was no good way of
# placing the constraint box. chose another non-pick env
env = "handle-press-v2"
envs_valid.remove("handle-press-v2")
while (env == "handle-press-v2") or (env in envs_valid_non_pick):
    env = random.sample(envs_non_pick, 1)
envs_valid += env
print(envs_valid)

