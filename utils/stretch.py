joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RSiteF", "LeftUpLeg", "LeftLeg",
                      "LeftFoot",
                      "LeftToeBase", "LSiteF", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                      "LeftForeArm",
                      "LeftHand", "LeftHandThumb", "LSiteT", "L_Wrist_End", "LSiteH", "RightShoulder", "RightArm",
                      "RightForeArm","RightHand", "RightHandThumb", "RSiteT", "R_Wrist_End", "RSiteH"]
joint_table = []

    for i, j in enumerate(joint_name):
        if j[:4] == 'Left':
            orig = joint_name.index("".join(["Right", j[4:]]))
        elif j[:5] == 'Right':
            orig = joint_name.index("".join(["Left", j[5:]]))
        elif j[0] == 'L':
            orig = joint_name.index("".join(["R", j[1:]]))
        elif j[0] == 'R':
            orig = joint_name.index("".join(["L", j[1:]]))
        else:
            orig = i
        joint_table.append(orig)

