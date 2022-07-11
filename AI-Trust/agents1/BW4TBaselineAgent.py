import enum
import json
import random
from typing import Dict

import numpy as np
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message

from bw4t.BW4TBrain import BW4TBrain

"""
    Explanation of each phase
    1 - agent will calculate the path to the closest door
    2 - agent will proceed to the door location via navigator
    3 - agent will perform the action open door when it arrived to the location
    4 - agent will inter the room
    5 - agent is initialized with standard settings unless required otherwise
        In this phase the agent gets will proceed to plan the path to the first door
    6 - agent will scan for goal blocks and if found will broadcast a message
    7 - (Phase A) agent will enter phase A after all rooms have been scanned and proceed to pick up the first goal block
    8 - (Phase B) agent arrives to the goal block and will check for edge conditions and if they are meet will actually
        pick up the block. agent will move to phase C or initiate phase A again
    9 - (Phase C) agent will proceed to the drop zone of the ghost block corresponding to the one is carrying
    10 - (Phase D) agent will drop the block and initiate phase A or phase C in case he is carrying 2 objects
    
    Algorithm stops when all blocks are dropped
"""
class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3,
    ENTER_ROOM = 4,
    INIT = 5,
    SCAN_FOR_OBJECTS_2 = 6,
    A = 7,
    B = 8,
    C = 9,
    D = 10


# class for representing ghost blocks
# contains: position, location, colour, shape
class GhostBlock:
    def __init__(self, pos, loc, colour, shape):
        self.pos = pos
        self.loc = loc
        self.colour = colour
        self.shape = shape

    # constructor from dictionary
    @classmethod
    def from_dict(cls, d):
        return cls(d['drop_zone_nr'], d['location'], d['visualization']['colour'], d['visualization']['shape'])

    # toString
    def __str__(self):
        return '{} {} {} {}'.format(self.pos, self.loc, self.colour, self.shape)

    # get the position of the ghost block
    def get_pos(self):
        return self.pos

    # get the location of the ghost block
    def get_loc(self):
        return self.loc

    # get the colour of the ghost block
    def get_colour(self):
        return self.colour

    # get the shape of the ghost block
    def get_shape(self):
        return self.shape

    # get the dictionary representation of the ghost block
    def to_dict(self):
        return {'drop_zone_nr': self.pos, 'location': self.loc,
                'visualization': {'colour': self.colour, 'shape': self.shape}}

    # equals method for ghost blocks
    def __eq__(self, other):
        return self.colour == other.colour and self.shape == other.shape


# class for representing the available blocks, extends GhostBlock
# contains: position, location, colour, shape, agent_id
class AvailableBlock(GhostBlock):
    def __init__(self, pos, loc, colour, shape, agent_id):
        super().__init__(pos, loc, colour, shape)
        self.agent_id = agent_id

    # toString
    def __str__(self):
        return '{} {} {} {} {}'.format(self.pos, self.loc, self.colour, self.shape, self.agent_id)

    # get the agent id of the available block
    def get_agent_id(self):
        return self.agent_id

    # get the dictionary representation of the available block
    def to_dict(self):
        return {'drop_zone_nr': self.pos, 'location': self.loc,
                'visualization': {'colour': self.colour, 'shape': self.shape},
                'agent_id': self.agent_id}

    # equals method for comparing one GhostBlock to AvailableBlock
    def equals(self, other):
        return self.colour == other.colour and self.shape == other.shape \
               and self.loc == other.loc and self.agent_id == other.agent_id

    # equals method for comparing one AvailableBlock to AvailableBlock
    def equals_available(self, other):
        return self.colour == other.colour and self.shape == other.shape

    @classmethod
    def from_dict(cls, d, loc, agent_id):
        return cls(0, loc, d['colour'], d['shape'], agent_id)


# class for representing the room
# contains: room_name, location, obj_id, visited
class Room:
    def __init__(self, room_name, location, obj_id):
        self.room_name = room_name
        self.location = location
        self.obj_id = obj_id
        self.visited = False

    # constructor from dictionary
    @classmethod
    def from_dict(cls, d):
        return cls(d['room_name'], d['location'], d['obj_id'])

    # toString
    def __str__(self):
        return '{} {} {} {}'.format(self.room_name, self.location, self.obj_id, self.visited)

    # get the room name
    def get_room_name(self):
        return self.room_name

    # get the location of the room
    def get_location(self):
        return self.location

    # get the object id of the room
    def get_obj_id(self):
        return self.obj_id

    # get the visited status of the room
    def get_visited(self):
        return self.visited

    # get the dictionary representation of the room
    def to_dict(self):
        return {'room_name': self.room_name, 'location': self.location, 'obj_id': self.obj_id}

    # equals method just for room name
    def __eq__(self, other):
        return self.room_name == other


class BaseLineAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.INIT
        self._teamMembers = []
        self._flag = True
        self._visited = []
        self._pickedUpBlocks = []
        # self._found_goal_blocks = []
        self._roomLocations = []
        self._ghostLocations = []
        self._availableLocations = []
        self._myMessages = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):

        return state

    def update_found_goal_block(self):
        for msg in self.received_messages:
            if "Found goal block " in msg.content:
                temp = msg.content[20: len(msg.content)]
                temp1 = temp.split("at location")
                dic = temp1[0].replace("'", "\"")
                loc = temp1[1]
                dic_new = json.loads(dic.replace("True", "true").replace("False", "false"))
                loc = loc.strip().replace("(", "").replace(")", "").split(',')
                location = (int(loc[0]), int(loc[1]))
                ava_loc = AvailableBlock.from_dict(dic_new, location, msg.from_id)
                if not (ava_loc in self._availableLocations):
                    self._availableLocations.append(ava_loc)

    def update_picked_goal_block(self):
        for msg in self.received_messages:
            if "Picking up goal block " in msg.content:
                temp = msg.content[22: len(msg.content)]
                temp1 = temp.split("at location")
                dic = temp1[0].replace("'", "\"")
                loc = temp1[1]
                dic_new = json.loads(dic.replace("True", "true").replace("False", "false"))
                loc = loc.strip().replace("(", "").replace(")", "").split(',')
                location = (int(loc[0]), int(loc[1]))
                ava_loc = AvailableBlock.from_dict(dic_new, location, msg.from_id)
                for block in self._ghostLocations:
                    if block.loc == ava_loc.loc and block.shape == ava_loc.shape and block.colour == ava_loc.colour:
                        self._ghostLocations.remove(block)
                        break

    def update_rooms_location(self):
        for msg in self.received_messages:
            if "Moving to " in msg.content:
                temp = msg.content[10:len(msg.content)]
                if temp in self._roomLocations:
                    self._roomLocations.remove(temp)

    def decide_on_bw4t_action(self, state: State):
        self.update_picked_goal_block()
        self.update_found_goal_block()
        self.update_rooms_location()

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:
            if Phase.INIT == self._phase:
                self.received_messages.clear()
                self._roomLocations.clear()
                temp = [door for door in state.values()
                        if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                for door in temp:
                    if not door['location'] in self._roomLocations:
                        self._roomLocations.append(Room.from_dict(door))

                self._ghostLocations = [GhostBlock.from_dict(ghostLoc) for ghostLoc in state.values()
                                        if 'is_drop_zone' in ghostLoc and 'GhostBlock' in ghostLoc['class_inheritance']
                                        and not ghostLoc['is_drop_zone']]

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                checkings = [room for room in self._roomLocations if not room.visited]
                if len(checkings) == 0:
                    self._phase = Phase.A
                    return None, {}
                #  pick a closed door
                # define the position of the door to be opened next
                next_door_position = -1;
                min_distance = 1000;
                # find the index of the closed door that is closest to the agent
                for i in range(len(self._roomLocations)):
                    if not self._roomLocations[i].visited:
                        x_difference = self._roomLocations[i].location[0] - state[agent_name]['location'][0]
                        y_difference = self._roomLocations[i].location[1] - state[agent_name]['location'][1]
                        distance = np.sqrt((x_difference ** 2) + (y_difference ** 2))
                        if (distance <= min_distance):
                            min_distance = distance
                            next_door_position = i
                self._roomLocations[next_door_position].visited = True
                self._door = self._roomLocations[next_door_position]
                # self._door = random.choice(closedDoors)
                doorLoc = self._door.location
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door.room_name, agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                for msg in self.received_messages:
                    if self._door.room_name in msg.content:
                        if msg.from_id > self.agent_id:
                            self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                            return None, {}

                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                # Open door
                self._phase = Phase.ENTER_ROOM
                self._sendMessage('Opening door of ' + self._door.room_name, agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door.obj_id}

            if Phase.ENTER_ROOM == self._phase:
                self._navigator.reset_full()
                location = self._door.location
                location = location[0], location[1] - 1
                self._navigator.add_waypoints([location])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                #self._sendMessage('Entering room ' + self._door.room_name, agent_name)
                if action != None:
                    return action, {}
                self._phase = Phase.SCAN_FOR_OBJECTS_2
                return action, {}

            if Phase.D == self._phase:
                self._navigator.reset_full()
                self._phase = Phase.A
                ghostLoc = [GhostBlock.from_dict(ghostLoc) for ghostLoc in state.values()
                            if 'is_drop_zone' in ghostLoc and 'GhostBlock' in ghostLoc['class_inheritance']
                            and not ghostLoc['is_drop_zone']]

                temp = state[self.agent_id]['location'][0], \
                       state[self.agent_id]['location'][1] + 1

                check_myMessages = False
                check_otherMessages = False
                # Check if current agent previously dropped a goal block on the precedent location
                for msg in self._myMessages:
                    if str(temp) in msg.content:
                        check_myMessages = True
                        break
                # Check if other agents previously dropped a goal block on the precedent location
                for msg in self.received_messages:
                    if str(temp) in msg.content and "Dropped goal block" in msg.content:
                        check_otherMessages = True
                        break
                if check_myMessages or check_otherMessages or ghostLoc[0].loc == \
                        state[self.agent_id]['location']:
                    pickMeUp2 = state[self.agent_id]['is_carrying'][0]['visualization']
                    del pickMeUp2['depth']
                    del pickMeUp2['opacity']
                    del pickMeUp2['visualize_from_center']
                    self._sendMessage(
                        'Dropped goal block ' + str(pickMeUp2)
                        + ' at location ' + str(state[self.agent_id]['location']), self.agent_id)
                    self._myMessages.append(Message(
                        content='Dropped goal block ' + str(pickMeUp2)
                                + ' at location ' + str(state[self.agent_id]['location']), from_id=self.agent_id))
                    return DropObject.__name__, {'object_id': state[self.agent_id]['is_carrying'][0]['obj_id']}

                self._phase = Phase.D
                return None, {}

            if Phase.C == self._phase:
                if len(state[self.agent_id]['is_carrying']) == 0:
                    self._phase = Phase.A
                    return None, {}

                self._navigator.reset_full()
                targetDrop = self._ghostLocations[0].loc
                self._navigator.add_waypoints([targetDrop])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._ghostLocations.remove(self._ghostLocations[0])
                self._phase = Phase.D

            if Phase.B == self._phase:
                targetPickUp = self._ghostLocations[0]
                self._navigator.reset_full()
                if state.get_of_type('CollectableBlock'):
                    pickMeUp = state.get_of_type('CollectableBlock')[0]
                    for availableBlock in state.get_of_type('CollectableBlock'):
                        # if availableBlock['visual']
                        temp = availableBlock['visualization']
                        if temp['shape'] == targetPickUp.shape and temp['colour']== targetPickUp.colour:
                            pickMeUp = availableBlock
                            break

                    newPickMe = AvailableBlock.from_dict(pickMeUp['visualization'], pickMeUp['location'],
                                                         self.agent_id)
                    if targetPickUp.colour == newPickMe.colour and targetPickUp.shape == newPickMe.shape:
                        pickMeUp2 = pickMeUp['visualization']
                        del pickMeUp2['depth']
                        del pickMeUp2['opacity']
                        del pickMeUp2['visualize_from_center']
                        self._sendMessage('Picking up goal block ' + str(pickMeUp2)
                                          + ' at location ' + str(newPickMe.loc), self.agent_id)
                        self._phase = Phase.C
                        return GrabObject.__name__, {'object_id': pickMeUp['obj_id']}
                    else:
                        self._ghostLocations.remove(self._ghostLocations[0])
                        self._phase = Phase.A
                        return None, {}
                else:
                    self._ghostLocations.remove(self._ghostLocations[0])
                    self._phase = Phase.A
                    return None, {}

            if Phase.A == self._phase:
                self._navigator.reset_full()
                if len(self._ghostLocations) == 0:
                    self._phase = Phase.INIT
                    return None, {}
                targetPickUp = self._ghostLocations[0]
                checkings = [block for block in self._availableLocations if block.shape == targetPickUp.shape
                             and block.colour == targetPickUp.colour]

                if len(checkings) > 0:
                    current_target = checkings[0]
                    for msg in self.received_messages:
                        if str(current_target.loc) in msg.content and "Picking up goal block" in msg.content:
                            if msg.from_id > self.agent_id:
                                self._ghostLocations.remove(self._ghostLocations[0])
                                self._phase = Phase.A
                                return None, {}

                    self._navigator.add_waypoints([current_target.loc])
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        return action, {}
                    self._phase = Phase.B  # TODO grab object
                else:
                    for msg in self.received_messages:
                        if not "Dropped goal block" in msg.content:
                            self.received_messages.remove(msg)
                    self._roomLocations.clear()
                    temp = [door for door in state.values()
                            if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                    for door in temp:
                        if not door['location'] in self._roomLocations:
                            self._roomLocations.append(Room.from_dict(door))
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                    return None, {}

            if Phase.SCAN_FOR_OBJECTS_2 == self._phase:
                self._navigator.reset_full()
                if state.get_of_type('CollectableBlock'):
                    pickMeUp = state.get_of_type('CollectableBlock')[0]
                    visPickMe = pickMeUp['visualization']

                    current_block = AvailableBlock.from_dict(visPickMe,
                                                             pickMeUp['location'],
                                                             self.agent_id)

                    #current_block.colour = '#000000'

                    checkings = [block for block in self._availableLocations
                                 if block.loc == current_block.loc and block.shape == current_block.shape
                                 and block.colour == current_block.colour]

                    if current_block in self._ghostLocations and len(checkings) == 0:
                        self._availableLocations.append(current_block)
                        pickMeUp2 = visPickMe
                        del pickMeUp2['depth']
                        del pickMeUp2['opacity']
                        del pickMeUp2['visualize_from_center']
                        self._sendMessage('Found goal block at ' + str(pickMeUp2)
                                          + ' at location ' +
                                          str(pickMeUp['location']), agent_name)

                location = self._door.location[0], self._door.location[1] - 1
                if state[self.agent_id]['location'] == location:
                    self._sendMessage('Searching through ' + self._door.room_name, agent_name)
                if self._flag:
                    if self.checkLeft(state):
                        target = state[self.agent_id]['location'][0] - 1, state[self.agent_id]['location'][1]

                    elif self.checkUp(state):
                        self._flag = not self._flag
                        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1

                    else:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                        return None, {}
                else:
                    if self.checkRight(state):
                        target = state[self.agent_id]['location'][0] + 1, state[self.agent_id]['location'][1]
                    elif self.checkUp(state):
                        self._flag = not self._flag
                        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1
                    else:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                        return None, {}

                self._navigator.add_waypoints([target])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                return action, {}

    def checkDirection(self, target, state):
        traverse_map = state.get_traverse_map()
        return traverse_map[target]

    def checkLeft(self, state):
        target = state[self.agent_id]['location'][0] - 1, state[self.agent_id]['location'][1]
        return self.checkDirection(target, state)

    def checkRight(self, state):
        target = state[self.agent_id]['location'][0] + 1, state[self.agent_id]['location'][1]
        return self.checkDirection(target, state)

    def checkUp(self, state):
        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1
        return self.checkDirection(target, state)

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def _trustBlief(self, member, received):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # You can change the default value to your preference
        default = 0.5
        trustBeliefs = {}
        for member in received.keys():
            trustBeliefs[member] = default
        for member in received.keys():
            for message in received[member]:
                if 'Found' in message and 'colour' not in message:
                    trustBeliefs[member] -= 0.1
                    break
        return trustBeliefs


class StrongAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.INIT
        self._teamMembers = []
        self._flag = True
        self._visited = []
        self._pickedUpBlocks = []
        # self._found_goal_blocks = []
        self._roomLocations = []
        self._ghostLocations = []
        self._availableLocations = []
        self.maxObjects = 2
        self.agentObjects = 0
        self._targetDrops = []
        self._myMessages = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):

        return state

    def update_found_goal_block(self):
        for msg in self.received_messages:
            if "Found goal block " in msg.content:
                temp = msg.content[20: len(msg.content)]
                temp1 = temp.split("at location")
                dic = temp1[0].replace("'", "\"")
                loc = temp1[1]
                dic_new = json.loads(dic.replace("True", "true").replace("False", "false"))
                loc = loc.strip().replace("(", "").replace(")", "").split(',')
                location = (int(loc[0]), int(loc[1]))
                ava_loc = AvailableBlock.from_dict(dic_new, location, msg.from_id)

                if not (ava_loc in self._availableLocations):
                    self._availableLocations.append(ava_loc)

    def update_picked_goal_block(self):
        for msg in self.received_messages:
            if "Picking up goal block " in msg.content:
                temp = msg.content[22: len(msg.content)]
                temp1 = temp.split("at location")
                dic = temp1[0].replace("'", "\"")
                loc = temp1[1]
                dic_new = json.loads(dic.replace("True", "true").replace("False", "false"))
                loc = loc.strip().replace("(", "").replace(")", "").split(',')
                location = (int(loc[0]), int(loc[1]))
                ava_loc = AvailableBlock.from_dict(dic_new, location, msg.from_id)
                for block in self._ghostLocations:
                    if block.loc == ava_loc.loc and block.shape == ava_loc.shape and block.colour == ava_loc.colour:
                        self._ghostLocations.remove(block)
                        break

    def update_rooms_location(self):
        for msg in self.received_messages:
            if "Moving to " in msg.content:
                temp = msg.content[10:len(msg.content)]
                if temp in self._roomLocations:
                    self._roomLocations.remove(temp)

    def decide_on_bw4t_action(self, state: State):
        self.update_picked_goal_block()
        self.update_found_goal_block()
        self.update_rooms_location()

        agent_name = state[self.agent_id]['obj_id']
        state[self.agent_id]['current_action_args'].update(max_objects=self.maxObjects)

        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:
            if Phase.INIT == self._phase:
                self.received_messages.clear()
                self._roomLocations.clear()
                temp = [door for door in state.values()
                        if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                for door in temp:
                    if not door['location'] in self._roomLocations:
                        self._roomLocations.append(Room.from_dict(door))

                self._ghostLocations = [GhostBlock.from_dict(ghostLoc) for ghostLoc in state.values()
                                        if 'is_drop_zone' in ghostLoc and 'GhostBlock' in ghostLoc['class_inheritance']
                                        and not ghostLoc['is_drop_zone']]

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                checkings = [room for room in self._roomLocations if not room.visited]
                if len(checkings) == 0:
                    self._phase = Phase.A
                    return None, {}
                #  pick a closed door
                # define the position of the door to be opened next
                next_door_position = -1;
                min_distance = 1000;
                # find the index of the closed door that is closest to the agent
                for i in range(len(self._roomLocations)):
                    if not self._roomLocations[i].visited:
                        x_difference = self._roomLocations[i].location[0] - state[agent_name]['location'][0]
                        y_difference = self._roomLocations[i].location[1] - state[agent_name]['location'][1]
                        distance = np.sqrt((x_difference ** 2) + (y_difference ** 2))
                        if (distance <= min_distance):
                            min_distance = distance
                            next_door_position = i
                self._roomLocations[next_door_position].visited = True
                self._door = self._roomLocations[next_door_position]
                # self._door = random.choice(closedDoors)
                doorLoc = self._door.location
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door.room_name, agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                for msg in self.received_messages:
                    if self._door.room_name in msg.content:
                        if msg.from_id > self.agent_id:
                            self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                            return None, {}

                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                # Open door
                self._phase = Phase.ENTER_ROOM
                self._sendMessage('Opening door of ' + self._door.room_name, agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door.obj_id}

            if Phase.ENTER_ROOM == self._phase:
                self._navigator.reset_full()
                location = self._door.location
                location = location[0], location[1] - 1
                self._navigator.add_waypoints([location])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                #self._sendMessage('Entering room ' + self._door.room_name, agent_name)
                if action != None:
                    return action, {}
                self._phase = Phase.SCAN_FOR_OBJECTS_2
                return action, {}

            if Phase.D == self._phase:
                self._navigator.reset_full()
                self._phase = Phase.A
                ghostLoc = [GhostBlock.from_dict(ghostLoc) for ghostLoc in state.values()
                            if 'is_drop_zone' in ghostLoc and 'GhostBlock' in ghostLoc['class_inheritance']
                            and not ghostLoc['is_drop_zone']]

                temp = state[self.agent_id]['location'][0], \
                       state[self.agent_id]['location'][1] + 1

                check_myMessages = False
                check_otherMessages = False
                # Check if current agent previously dropped a goal block on the precedent location
                for msg in self._myMessages:
                    if str(temp) in msg.content:
                        check_myMessages = True
                        break
                # Check if other agents previously dropped a goal block on the precedent location
                for msg in self.received_messages:
                    if str(temp) in msg.content and "Dropped goal block" in msg.content:
                        check_otherMessages = True
                        break
                if check_myMessages or check_otherMessages or ghostLoc[0].loc == \
                        state[self.agent_id]['location']:
                    pickMeUp2 = state[self.agent_id]['is_carrying'][0]['visualization']
                    del pickMeUp2['depth']
                    del pickMeUp2['opacity']
                    del pickMeUp2['visualize_from_center']
                    self._sendMessage(
                        'Dropped goal block ' + str(pickMeUp2)
                        + ' at location ' + str(state[self.agent_id]['location']), self.agent_id)
                    self._myMessages.append(Message(
                        content='Dropped goal block ' + str(pickMeUp2)
                                + ' at location ' + str(state[self.agent_id]['location']), from_id=self.agent_id))
                    if self.agentObjects > 1:
                        self._phase = Phase.C
                        self.agentObjects -= 1
                        self._targetDrops.__delitem__(0)
                    return DropObject.__name__, {'object_id': state[self.agent_id]['is_carrying'][0]['obj_id']}


                self._phase = Phase.D
                return None, {}

            if Phase.C == self._phase:
                check_drop = False
                if len(state[self.agent_id]['is_carrying']) == 0:
                    self._phase = Phase.A
                    return None, {}

                if len(state[self.agent_id]['is_carrying']) > 0 and len(self._ghostLocations) == 0:
                    check_drop = True
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._targetDrops[0]])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if action != None:
                    return action, {}
                if not check_drop:
                    self._ghostLocations.remove(self._ghostLocations[0])
                self._phase = Phase.D

            if Phase.B == self._phase:
                targetPickUp = self._ghostLocations[0]
                self._navigator.reset_full()
                if state.get_of_type('CollectableBlock'):

                    pickMeUp =  state.get_of_type('CollectableBlock')[0]
                    for availableBlock in state.get_of_type('CollectableBlock'):
                        # if availableBlock['visual']
                        temp = availableBlock['visualization']
                        if temp['shape'] == targetPickUp.shape and temp['colour'] == targetPickUp.colour:
                            pickMeUp = availableBlock
                            break
                    newPickMe = AvailableBlock.from_dict(pickMeUp['visualization'], pickMeUp['location'],
                                                         self.agent_id)
                    if targetPickUp.colour == newPickMe.colour and targetPickUp.shape == newPickMe.shape:
                        pickMeUp2 = pickMeUp['visualization']
                        del pickMeUp2['depth']
                        del pickMeUp2['opacity']
                        del pickMeUp2['visualize_from_center']
                        self._sendMessage('Picking up goal block ' + str(pickMeUp2)
                                          + ' at location ' + str(newPickMe.loc), self.agent_id)

                        self.agentObjects += 1
                        if self.agentObjects < 2 and len(self._ghostLocations) == 0:
                            self._phase = Phase.C
                        elif self.agentObjects < 2 and len(self._ghostLocations) < 2:
                            self._phase = Phase.C
                        elif self.agentObjects == 2:
                            self._phase = Phase.C
                        else:
                            self._phase = Phase.A
                        self._targetDrops.append(self._ghostLocations[0].loc)

                        return GrabObject.__name__, {'object_id': pickMeUp['obj_id']}
                    else:
                        self._ghostLocations.remove(self._ghostLocations[0])
                        self._phase = Phase.A
                        return None, {}
                else:
                    self._ghostLocations.remove(self._ghostLocations[0])
                    self._phase = Phase.A
                    return None, {}

            if Phase.A == self._phase:
                self._navigator.reset_full()
                if len(self._ghostLocations) == 0:
                    self._phase = Phase.INIT
                    return None, {}
                targetPickUp = self._ghostLocations[0]
                checkings = [block for block in self._availableLocations if block.shape == targetPickUp.shape
                             and block.colour == targetPickUp.colour]

                if self.agentObjects > 0:
                    for msg in self.received_messages:
                        if "Dropped goal block" in msg.content:
                            self._phase = Phase.C
                            return None, {}

                if len(checkings) > 0:
                    current_target = checkings[0]
                    for msg in self.received_messages:
                        if str(current_target.loc) in msg.content and "Picking up goal block" in msg.content:
                            if msg.from_id > self.agent_id and msg.from_id != self.agent_id:
                                self._ghostLocations.remove(self._ghostLocations[0])
                                self._phase = Phase.A
                                return None, {}

                    self._navigator.add_waypoints([current_target.loc])
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        return action, {}
                    self._phase = Phase.B  # TODO grab object
                else:
                    for msg in self.received_messages:
                        if not "Dropped goal block" in msg.content:
                            self.received_messages.remove(msg)
                    self._roomLocations.clear()
                    temp = [door for door in state.values()
                            if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                    for door in temp:
                        if not door['location'] in self._roomLocations:
                            self._roomLocations.append(Room.from_dict(door))
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                    return None, {}

            if Phase.SCAN_FOR_OBJECTS_2 == self._phase:
                self._navigator.reset_full()
                if state.get_of_type('CollectableBlock'):
                    pickMeUp = state.get_of_type('CollectableBlock')[0]
                    visPickMe = pickMeUp['visualization']

                    current_block = AvailableBlock.from_dict(visPickMe,
                                                             pickMeUp['location'],
                                                             self.agent_id)

                    checkings = [block for block in self._availableLocations
                                 if block.loc == current_block.loc and block.shape == current_block.shape
                                 and block.colour == current_block.colour]
                    if current_block in self._ghostLocations and len(checkings) == 0:
                        self._availableLocations.append(current_block)
                        pickMeUp2 = visPickMe
                        del pickMeUp2['depth']
                        del pickMeUp2['opacity']
                        del pickMeUp2['visualize_from_center']
                        self._sendMessage('Found goal block at ' + str(pickMeUp2)
                                          + ' at location ' +
                                          str(pickMeUp['location']), agent_name)

                location = self._door.location[0], self._door.location[1] - 1
                if state[self.agent_id]['location'] == location:
                    self._sendMessage('Searching through ' + self._door.room_name, agent_name)
                if self._flag:
                    if self.checkLeft(state):
                        target = state[self.agent_id]['location'][0] - 1, state[self.agent_id]['location'][1]

                    elif self.checkUp(state):
                        self._flag = not self._flag
                        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1

                    else:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                        return None, {}
                else:
                    if self.checkRight(state):
                        target = state[self.agent_id]['location'][0] + 1, state[self.agent_id]['location'][1]
                    elif self.checkUp(state):
                        self._flag = not self._flag
                        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1
                    else:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                        return None, {}

                self._navigator.add_waypoints([target])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                return action, {}

    def checkDirection(self, target, state):
        traverse_map = state.get_traverse_map()
        return traverse_map[target]

    def checkLeft(self, state):
        target = state[self.agent_id]['location'][0] - 1, state[self.agent_id]['location'][1]
        return self.checkDirection(target, state)

    def checkRight(self, state):
        target = state[self.agent_id]['location'][0] + 1, state[self.agent_id]['location'][1]
        return self.checkDirection(target, state)

    def checkUp(self, state):
        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1
        return self.checkDirection(target, state)

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def _trustBlief(self, member, received):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # You can change the default value to your preference
        default = 0.5
        trustBeliefs = {}
        for member in received.keys():
            trustBeliefs[member] = default
        for member in received.keys():
            for message in received[member]:
                if 'Found' in message and 'colour' not in message:
                    trustBeliefs[member] -= 0.1
                    break
        return trustBeliefs


class ColourblindAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.INIT
        self._teamMembers = []
        self._flag = True
        self._visited = []
        self._pickedUpBlocks = []
        # self._found_goal_blocks = []
        self._roomLocations = []
        self._ghostLocations = []
        self._availableLocations = []
        self._myMessages = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):

        return state

    def update_found_goal_block(self):
        for msg in self.received_messages:
            if "Found goal block " in msg.content:
                temp = msg.content[20: len(msg.content)]
                temp1 = temp.split("at location")
                dic = temp1[0].replace("'", "\"")
                loc = temp1[1]
                dic_new = json.loads(dic.replace("True", "true").replace("False", "false"))
                loc = loc.strip().replace("(", "").replace(")", "").split(',')
                location = (int(loc[0]), int(loc[1]))
                ava_loc = AvailableBlock.from_dict(dic_new, location, msg.from_id)
                ava_loc.colour = '#000000'
                if not (ava_loc in self._availableLocations):
                    self._availableLocations.append(ava_loc)

    def update_picked_goal_block(self):
        for msg in self.received_messages:
            if "Picking up goal block " in msg.content:
                temp = msg.content[22: len(msg.content)]
                temp1 = temp.split("at location")
                dic = temp1[0].replace("'", "\"")
                loc = temp1[1]
                dic_new = json.loads(dic.replace("True", "true").replace("False", "false"))
                loc = loc.strip().replace("(", "").replace(")", "").split(',')
                location = (int(loc[0]), int(loc[1]))
                ava_loc = AvailableBlock.from_dict(dic_new, location, msg.from_id)
                ava_loc.colour = '#000000'

                for block in self._ghostLocations:
                    if block.loc == ava_loc.loc and block.shape == ava_loc.shape and block.colour == ava_loc.colour:
                        self._ghostLocations.remove(block)
                        break

    def update_rooms_location(self):
        for msg in self.received_messages:
            if "Moving to " in msg.content:
                temp = msg.content[10:len(msg.content)]

                if temp in self._roomLocations:
                    self._roomLocations.remove(temp)

    def decide_on_bw4t_action(self, state: State):
        self.update_picked_goal_block()
        self.update_found_goal_block()
        self.update_rooms_location()

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:
            if Phase.INIT == self._phase:
                self.received_messages.clear()
                self._roomLocations.clear()
                temp = [door for door in state.values()
                        if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                for door in temp:
                    if not door['location'] in self._roomLocations:
                        self._roomLocations.append(Room.from_dict(door))

                self._ghostLocations = [GhostBlock.from_dict(ghostLoc) for ghostLoc in state.values()
                                        if 'is_drop_zone' in ghostLoc and 'GhostBlock' in ghostLoc['class_inheritance']
                                        and not ghostLoc['is_drop_zone']]
                for i in range(len(self._ghostLocations)):
                    self._ghostLocations[i].colour = '#000000'
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                checkings = [room for room in self._roomLocations if not room.visited]
                if len(checkings) == 0:
                    self._phase = Phase.A
                    return None, {}
                #  pick a closed door
                # define the position of the door to be opened next
                next_door_position = -1;
                min_distance = 1000;
                # find the index of the closed door that is closest to the agent
                for i in range(len(self._roomLocations)):
                    if not self._roomLocations[i].visited:
                        x_difference = self._roomLocations[i].location[0] - state[agent_name]['location'][0]
                        y_difference = self._roomLocations[i].location[1] - state[agent_name]['location'][1]
                        distance = np.sqrt((x_difference ** 2) + (y_difference ** 2))
                        if (distance <= min_distance):
                            min_distance = distance
                            next_door_position = i
                self._roomLocations[next_door_position].visited = True
                self._door = self._roomLocations[next_door_position]
                # self._door = random.choice(closedDoors)
                doorLoc = self._door.location
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door.room_name, agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                for msg in self.received_messages:
                    if self._door.room_name in msg.content:
                        if msg.from_id > self.agent_id:
                            self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                            return None, {}

                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                # Open door
                self._phase = Phase.ENTER_ROOM
                self._sendMessage('Opening door of ' + self._door.room_name, agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door.obj_id}

            if Phase.ENTER_ROOM == self._phase:
                self._navigator.reset_full()
                location = self._door.location
                location = location[0], location[1] - 1
                self._navigator.add_waypoints([location])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                #self._sendMessage('Entering room ' + self._door.room_name, agent_name)
                if action != None:
                    return action, {}
                self._phase = Phase.SCAN_FOR_OBJECTS_2
                return action, {}

            if Phase.D == self._phase:
                self._navigator.reset_full()
                self._phase = Phase.A
                ghostLoc = [GhostBlock.from_dict(ghostLoc) for ghostLoc in state.values()
                            if 'is_drop_zone' in ghostLoc and 'GhostBlock' in ghostLoc['class_inheritance']
                            and not ghostLoc['is_drop_zone']]

                temp = state[self.agent_id]['location'][0], \
                       state[self.agent_id]['location'][1] + 1

                check_myMessages = False
                check_otherMessages = False
                # Check if current agent previously dropped a goal block on the precedent location
                for msg in self._myMessages:
                    if str(temp) in msg.content:
                        check_myMessages = True
                        break
                # Check if other agents previously dropped a goal block on the precedent location
                for msg in self.received_messages:
                    if str(temp) in msg.content and "Dropped goal block" in msg.content:
                        check_otherMessages = True
                        break
                if check_myMessages or check_otherMessages or ghostLoc[0].loc == \
                        state[self.agent_id]['location']:
                    pickMeUp2 = state[self.agent_id]['is_carrying'][0]['visualization']
                    del pickMeUp2['depth']
                    del pickMeUp2['opacity']
                    del pickMeUp2['visualize_from_center']
                    self._sendMessage(
                        'Dropped goal block ' + str(pickMeUp2)
                        + ' at location ' + str(state[self.agent_id]['location']), self.agent_id)
                    self._myMessages.append(Message(
                        content='Dropped goal block ' + str(pickMeUp2)
                                + ' at location ' + str(state[self.agent_id]['location']), from_id=self.agent_id))
                    return DropObject.__name__, {'object_id': state[self.agent_id]['is_carrying'][0]['obj_id']}

                self._phase = Phase.D
                return None, {}

            if Phase.C == self._phase:
                if len(state[self.agent_id]['is_carrying']) == 0:
                    self._phase = Phase.A
                    return None, {}

                self._navigator.reset_full()
                targetDrop = self._ghostLocations[0].loc
                self._navigator.add_waypoints([targetDrop])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._ghostLocations.remove(self._ghostLocations[0])
                self._phase = Phase.D

            if Phase.B == self._phase:
                targetPickUp = self._ghostLocations[0]
                self._navigator.reset_full()
                if state.get_of_type('CollectableBlock'):
                    pickMeUp = state.get_of_type('CollectableBlock')[0]
                    for availableBlock in state.get_of_type('CollectableBlock'):
                        # if availableBlock['visual']
                        temp = availableBlock['visualization']
                        temp['colour'] = '#000000'
                        if temp['shape'] == targetPickUp.shape and temp['colour'] == targetPickUp.colour:
                            pickMeUp = availableBlock
                            break

                    pickMeUp['visualization'].update(colour = '#000000')
                    newPickMe = AvailableBlock.from_dict(pickMeUp['visualization'], pickMeUp['location'],
                                                         self.agent_id)
                    if targetPickUp.colour == newPickMe.colour and targetPickUp.shape == newPickMe.shape:
                        pickMeUp2 = pickMeUp['visualization']
                        del pickMeUp2['depth']
                        del pickMeUp2['opacity']
                        del pickMeUp2['visualize_from_center']
                        self._sendMessage('Picking up goal block ' + str(pickMeUp2)
                                          + ' at location ' + str(newPickMe.loc), self.agent_id)
                        self._phase = Phase.C
                        return GrabObject.__name__, {'object_id': pickMeUp['obj_id']}
                    else:
                        self._ghostLocations.remove(self._ghostLocations[0])
                        self._phase = Phase.A
                        return None, {}
                else:
                    self._ghostLocations.remove(self._ghostLocations[0])
                    self._phase = Phase.A
                    return None, {}

            if Phase.A == self._phase:
                self._navigator.reset_full()
                if len(self._ghostLocations) == 0:
                    self._phase = Phase.INIT
                    return None, {}
                targetPickUp = self._ghostLocations[0]
                checkings = [block for block in self._availableLocations if block.shape == targetPickUp.shape
                             and block.colour == targetPickUp.colour]

                if len(checkings) > 0:
                    current_target = checkings[0]
                    for msg in self.received_messages:
                        if str(current_target.loc) in msg.content and "Picking up goal block" in msg.content:
                            if msg.from_id > self.agent_id:
                                self._ghostLocations.remove(self._ghostLocations[0])
                                self._phase = Phase.A
                                return None, {}

                    self._navigator.add_waypoints([current_target.loc])
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        return action, {}
                    self._phase = Phase.B  # TODO grab object
                else:
                    for msg in self.received_messages:
                        if not "Dropped goal block" in msg.content:
                            self.received_messages.remove(msg)
                    self._roomLocations.clear()
                    temp = [door for door in state.values()
                            if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                    for door in temp:
                        if not door['location'] in self._roomLocations:
                            self._roomLocations.append(Room.from_dict(door))
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                    return None, {}

            if Phase.SCAN_FOR_OBJECTS_2 == self._phase:
                self._navigator.reset_full()
                if state.get_of_type('CollectableBlock'):
                    pickMeUp = state.get_of_type('CollectableBlock')[0]
                    visPickMe = pickMeUp['visualization']

                    current_block = AvailableBlock.from_dict(visPickMe,
                                                             pickMeUp['location'],
                                                             self.agent_id)
                    current_block.colour = '#000000'
                    checkings = [block for block in self._availableLocations
                                 if block.loc == current_block.loc and block.shape == current_block.shape
                                 and block.colour == current_block.colour]

                    if current_block in self._ghostLocations and len(checkings) == 0:
                        self._availableLocations.append(current_block)
                        visPickMe['colour'] = '#000000'
                        pickMeUp2 = visPickMe
                        del pickMeUp2['depth']
                        del pickMeUp2['opacity']
                        del pickMeUp2['visualize_from_center']
                        self._sendMessage('Found goal block at ' + str(pickMeUp2)
                                          + ' at location ' +
                                          str(pickMeUp['location']), agent_name)

                location = self._door.location[0], self._door.location[1] - 1
                if state[self.agent_id]['location'] == location:
                    self._sendMessage('Searching through ' + self._door.room_name, agent_name)
                if self._flag:
                    if self.checkLeft(state):
                        target = state[self.agent_id]['location'][0] - 1, state[self.agent_id]['location'][1]

                    elif self.checkUp(state):
                        self._flag = not self._flag
                        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1

                    else:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                        return None, {}
                else:
                    if self.checkRight(state):
                        target = state[self.agent_id]['location'][0] + 1, state[self.agent_id]['location'][1]
                    elif self.checkUp(state):
                        self._flag = not self._flag
                        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1
                    else:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                        return None, {}

                self._navigator.add_waypoints([target])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                return action, {}

    def checkDirection(self, target, state):
        traverse_map = state.get_traverse_map()
        return traverse_map[target]

    def checkLeft(self, state):
        target = state[self.agent_id]['location'][0] - 1, state[self.agent_id]['location'][1]
        return self.checkDirection(target, state)

    def checkRight(self, state):
        target = state[self.agent_id]['location'][0] + 1, state[self.agent_id]['location'][1]
        return self.checkDirection(target, state)

    def checkUp(self, state):
        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1
        return self.checkDirection(target, state)

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def _trustBlief(self, member, received):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # You can change the default value to your preference
        default = 0.5
        trustBeliefs = {}
        for member in received.keys():
            trustBeliefs[member] = default
        for member in received.keys():
            for message in received[member]:
                if 'Found' in message and 'colour' not in message:
                    trustBeliefs[member] -= 0.1
                    break
        return trustBeliefs


class LiarAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.INIT
        self._teamMembers = []
        self._flag = True
        self._visited = []
        self._pickedUpBlocks = []
        # self._found_goal_blocks = []
        self._roomLocations = []
        self._ghostLocations = []
        self._availableLocations = []
        self._myMessages = []
    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):

        return state

    def decide_to_lie(self):
        choice = [0,1]
        final_choice = random.choices(population = choice, cum_weights = [80,20], k = 1)
        if final_choice[0] == 0:
            return True
        return False

    def update_found_goal_block(self):
        for msg in self.received_messages:
            if "Found goal block " in msg.content:
                temp = msg.content[20: len(msg.content)]
                temp1 = temp.split("at location")
                dic = temp1[0].replace("'", "\"")
                loc = temp1[1]
                dic_new = json.loads(dic.replace("True", "true").replace("False", "false"))
                loc = loc.strip().replace("(", "").replace(")", "").split(',')
                location = (int(loc[0]), int(loc[1]))
                ava_loc = AvailableBlock.from_dict(dic_new, location, msg.from_id)

                if not (ava_loc in self._availableLocations):
                    self._availableLocations.append(ava_loc)

    def update_picked_goal_block(self):
        for msg in self.received_messages:
            if "Picking up goal block " in msg.content:
                temp = msg.content[22: len(msg.content)]
                temp1 = temp.split("at location")
                dic = temp1[0].replace("'", "\"")
                loc = temp1[1]
                dic_new = json.loads(dic.replace("True", "true").replace("False", "false"))
                loc = loc.strip().replace("(", "").replace(")", "").split(',')
                location = (int(loc[0]), int(loc[1]))
                ava_loc = AvailableBlock.from_dict(dic_new, location, msg.from_id)

                for block in self._ghostLocations:
                    if block.loc == ava_loc.loc and block.shape == ava_loc.shape and block.colour == ava_loc.colour:
                        self._ghostLocations.remove(block)
                        break


    def update_rooms_location(self):
        for msg in self.received_messages:
            if "Moving to " in msg.content:
                temp = msg.content[10:len(msg.content)]
                if temp in self._roomLocations:
                    self._roomLocations.remove(temp)

    def decide_on_bw4t_action(self, state: State):
        self.update_picked_goal_block()
        self.update_found_goal_block()
        self.update_rooms_location()

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:
            if Phase.INIT == self._phase:
                self.received_messages.clear()
                self._roomLocations.clear()
                temp = [door for door in state.values()
                        if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                for door in temp:
                    if not door['location'] in self._roomLocations:
                        self._roomLocations.append(Room.from_dict(door))

                self._ghostLocations = [GhostBlock.from_dict(ghostLoc) for ghostLoc in state.values()
                                        if 'is_drop_zone' in ghostLoc and 'GhostBlock' in ghostLoc['class_inheritance']
                                        and not ghostLoc['is_drop_zone']]

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                checkings = [room for room in self._roomLocations if not room.visited]
                if len(checkings) == 0:
                    self._phase = Phase.A
                    return None, {}
                #  pick a closed door
                # define the position of the door to be opened next
                next_door_position = -1;
                min_distance = 1000;
                # find the index of the closed door that is closest to the agent
                for i in range(len(self._roomLocations)):
                    if not self._roomLocations[i].visited:
                        x_difference = self._roomLocations[i].location[0] - state[agent_name]['location'][0]
                        y_difference = self._roomLocations[i].location[1] - state[agent_name]['location'][1]
                        distance = np.sqrt((x_difference ** 2) + (y_difference ** 2))
                        if (distance <= min_distance):
                            min_distance = distance
                            next_door_position = i
                self._roomLocations[next_door_position].visited = True
                self._door = self._roomLocations[next_door_position]
                # self._door = random.choice(closedDoors)
                doorLoc = self._door.location
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                door_name = self._door.room_name
                if self.decide_to_lie():
                    # Choose any other room name that is different from the current one to lie
                    # Also choose room that is not visited
                    diff_names = [room.room_name for room in self._roomLocations if room.room_name != door_name and not room.visited]
                    if len(diff_names) != 0:
                        door_name = diff_names[0]
                self._sendMessage('Moving to ' + door_name, agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                for msg in self.received_messages:
                    if self._door.room_name in msg.content:
                        if msg.from_id > self.agent_id:
                            self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                            return None, {}

                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                # Open door
                self._phase = Phase.ENTER_ROOM
                return OpenDoorAction.__name__, {'object_id': self._door.obj_id}

            if Phase.ENTER_ROOM == self._phase:
                self._navigator.reset_full()
                location = self._door.location
                location = location[0], location[1] - 1
                self._navigator.add_waypoints([location])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                #self._sendMessage('Entering room ' + self._door.room_name, agent_name)
                if action != None:
                    return action, {}
                self._phase = Phase.SCAN_FOR_OBJECTS_2
                return action, {}

            if Phase.D == self._phase:
                self._navigator.reset_full()
                self._phase = Phase.A
                ghostLoc = [GhostBlock.from_dict(ghostLoc) for ghostLoc in state.values()
                            if 'is_drop_zone' in ghostLoc and 'GhostBlock' in ghostLoc['class_inheritance']
                            and not ghostLoc['is_drop_zone']]

                temp = state[self.agent_id]['location'][0], \
                       state[self.agent_id]['location'][1] + 1
                check_myMessages = False
                check_otherMessages = False
                # Check if current agent previously dropped a goal block on the precedent location
                for msg in self._myMessages:
                    if str(temp) in msg.content:
                        check_myMessages = True
                        break
                # Check if other agents previously dropped a goal block on the precedent location
                for msg in self.received_messages:
                    if str(temp) in msg.content and "Dropped goal block" in msg.content:
                        check_otherMessages = True
                        break
                if check_myMessages or check_otherMessages or ghostLoc[0].loc == \
                        state[self.agent_id]['location']:
                    pickMeUp2 = state[self.agent_id]['is_carrying'][0]['visualization']
                    del pickMeUp2['depth']
                    del pickMeUp2['opacity']
                    del pickMeUp2['visualize_from_center']
                    self._sendMessage(
                        'Dropped goal block ' + str(pickMeUp2)
                        + ' at location ' + str(state[self.agent_id]['location']), self.agent_id)
                    self._myMessages.append(Message(
                        content='Dropped goal block ' + str(pickMeUp2)
                                + ' at location ' + str(state[self.agent_id]['location']), from_id=self.agent_id))
                    return DropObject.__name__, {'object_id': state[self.agent_id]['is_carrying'][0]['obj_id']}

                self._phase = Phase.D
                return None, {}

            if Phase.C == self._phase:
                if len(state[self.agent_id]['is_carrying']) == 0:
                    self._phase = Phase.A
                    return None, {}

                self._navigator.reset_full()
                targetDrop = self._ghostLocations[0].loc
                self._navigator.add_waypoints([targetDrop])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._ghostLocations.remove(self._ghostLocations[0])
                self._phase = Phase.D

            if Phase.B == self._phase:
                targetPickUp = self._ghostLocations[0]
                self._navigator.reset_full()
                if state.get_of_type('CollectableBlock'):
                    pickMeUp = state.get_of_type('CollectableBlock')[0]
                    for availableBlock in state.get_of_type('CollectableBlock'):
                        # if availableBlock['visual']
                        temp = availableBlock['visualization']
                        if temp['shape'] == targetPickUp.shape and temp['colour'] == targetPickUp.colour:
                            pickMeUp = availableBlock
                            break
                    newPickMe = AvailableBlock.from_dict(pickMeUp['visualization'], pickMeUp['location'],
                                                         self.agent_id)
                    if targetPickUp.colour == newPickMe.colour and targetPickUp.shape == newPickMe.shape:
                        final_location = newPickMe.loc
                        pickMeUp2 = pickMeUp['visualization']
                        del pickMeUp2['depth']
                        del pickMeUp2['opacity']
                        del pickMeUp2['visualize_from_center']
                        self._sendMessage('Picking up goal block ' + str(pickMeUp2)
                                          + ' at location ' + str(final_location), self.agent_id)
                        self._phase = Phase.C
                        return GrabObject.__name__, {'object_id': pickMeUp['obj_id']}
                    else:
                        self._ghostLocations.remove(self._ghostLocations[0])
                        self._phase = Phase.A
                        return None, {}
                else:
                    self._ghostLocations.remove(self._ghostLocations[0])
                    self._phase = Phase.A
                    return None, {}

            if Phase.A == self._phase:
                self._navigator.reset_full()
                if len(self._ghostLocations) == 0:
                    self._phase = Phase.INIT
                    return None, {}
                targetPickUp = self._ghostLocations[0]
                checkings = [block for block in self._availableLocations if block.shape == targetPickUp.shape
                             and block.colour == targetPickUp.colour]

                if len(checkings) > 0:
                    current_target = checkings[0]
                    for msg in self.received_messages:
                        if str(current_target.loc) in msg.content and "Picking up goal block" in msg.content:
                            if msg.from_id > self.agent_id:
                                self._ghostLocations.remove(self._ghostLocations[0])
                                self._phase = Phase.A
                                return None, {}

                    self._navigator.add_waypoints([current_target.loc])
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        return action, {}
                    self._phase = Phase.B  # TODO grab object
                else:
                    for msg in self.received_messages:
                        if not "Dropped goal block" in msg.content:
                            self.received_messages.remove(msg)
                    self._roomLocations.clear()
                    temp = [door for door in state.values()
                            if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                    for door in temp:
                        if not door['location'] in self._roomLocations:
                            self._roomLocations.append(Room.from_dict(door))
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                    return None, {}

            if Phase.SCAN_FOR_OBJECTS_2 == self._phase:
                self._navigator.reset_full()
                if state.get_of_type('CollectableBlock'):
                    pickMeUp = state.get_of_type('CollectableBlock')[0]
                    visPickMe = pickMeUp['visualization']

                    current_block = AvailableBlock.from_dict(visPickMe,
                                                             pickMeUp['location'],
                                                             self.agent_id)

                    checkings = [block for block in self._availableLocations
                                 if block.loc == current_block.loc and block.shape == current_block.shape
                                 and block.colour == current_block.colour]

                    if current_block in self._ghostLocations and len(checkings) == 0:
                        self._availableLocations.append(current_block)

                        #Make the agent lie about shapes and colors
                        if self.decide_to_lie():

                            poss_shapes = [0,1,2]
                            poss_colors = ['#0008ff', '#0dff00', '#ff1500']

                            for shape in poss_shapes:
                                if visPickMe['shape'] != shape:
                                    visPickMe['shape'] = shape
                                    break
                            for color in poss_colors:
                                if visPickMe['colour'] != color:
                                    visPickMe['colour'] = color.replace('"', '')
                                    break
                        pickMeUp2 = visPickMe
                        del pickMeUp2['depth']
                        del pickMeUp2['opacity']
                        del pickMeUp2['visualize_from_center']
                        self._sendMessage('Found goal block at ' + str(pickMeUp2)
                                          + ' at location ' +
                                          str(pickMeUp['location']), agent_name)

                location = self._door.location[0], self._door.location[1] - 1
                if state[self.agent_id]['location'] == location:
                    self._sendMessage('Searching through ' + self._door.room_name, agent_name)
                if self._flag:
                    if self.checkLeft(state):
                        target = state[self.agent_id]['location'][0] - 1, state[self.agent_id]['location'][1]

                    elif self.checkUp(state):
                        self._flag = not self._flag
                        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1

                    else:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                        return None, {}
                else:
                    if self.checkRight(state):
                        target = state[self.agent_id]['location'][0] + 1, state[self.agent_id]['location'][1]
                    elif self.checkUp(state):
                        self._flag = not self._flag
                        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1
                    else:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                        return None, {}

                self._navigator.add_waypoints([target])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                return action, {}

    def checkDirection(self, target, state):
        traverse_map = state.get_traverse_map()
        return traverse_map[target]

    def checkLeft(self, state):
        target = state[self.agent_id]['location'][0] - 1, state[self.agent_id]['location'][1]
        return self.checkDirection(target, state)

    def checkRight(self, state):
        target = state[self.agent_id]['location'][0] + 1, state[self.agent_id]['location'][1]
        return self.checkDirection(target, state)

    def checkUp(self, state):
        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1
        return self.checkDirection(target, state)

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def _trustBlief(self, member, received):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # You can change the default value to your preference
        default = 0.5
        trustBeliefs = {}
        for member in received.keys():
            trustBeliefs[member] = default
        for member in received.keys():
            for message in received[member]:
                if 'Found' in message and 'colour' not in message:
                    trustBeliefs[member] -= 0.1
                    break
        return trustBeliefs


class LazyAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.INIT
        self._teamMembers = []
        self._flag = True
        self._visited = []
        self._pickedUpBlocks = []
        # self._found_goal_blocks = []
        self._roomLocations = []
        self._ghostLocations = []
        self._availableLocations = []
        self._quit = False
        self._myMessages = []
    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):

        return state

    def update_found_goal_block(self):
        for msg in self.received_messages:
            if "Found goal block " in msg.content:
                temp = msg.content[20: len(msg.content)]
                temp1 = temp.split("at location")
                dic = temp1[0].replace("'", "\"")
                loc = temp1[1]
                dic_new = json.loads(dic.replace("True", "true").replace("False", "false"))
                loc = loc.strip().replace("(", "").replace(")", "").split(',')
                location = (int(loc[0]), int(loc[1]))
                ava_loc = AvailableBlock.from_dict(dic_new, location, msg.from_id)

                if not (ava_loc in self._availableLocations):
                    self._availableLocations.append(ava_loc)

    def update_picked_goal_block(self):
        for msg in self.received_messages:
            if "Picking up goal block " in msg.content:
                temp = msg.content[22: len(msg.content)]
                temp1 = temp.split("at location")
                dic = temp1[0].replace("'", "\"")
                loc = temp1[1]
                dic_new = json.loads(dic.replace("True", "true").replace("False", "false"))
                loc = loc.strip().replace("(", "").replace(")", "").split(',')
                location = (int(loc[0]), int(loc[1]))
                ava_loc = AvailableBlock.from_dict(dic_new, location, msg.from_id)
                for block in self._ghostLocations:
                    if block.loc == ava_loc.loc and block.shape == ava_loc.shape and block.colour == ava_loc.colour:
                        self._ghostLocations.remove(block)
                        break

    def update_rooms_location(self):
        for msg in self.received_messages:
            if "Moving to " in msg.content:
                temp = msg.content[10:len(msg.content)]
                if temp in self._roomLocations:
                    self._roomLocations.remove(temp)

    def decide_on_bw4t_action(self, state: State):
        self.update_picked_goal_block()
        self.update_found_goal_block()
        self.update_rooms_location()

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:
            if Phase.INIT == self._phase:
                self.received_messages.clear()
                self._roomLocations.clear()
                temp = [door for door in state.values()
                        if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                for door in temp:
                    if not door['location'] in self._roomLocations:
                        self._roomLocations.append(Room.from_dict(door))

                self._ghostLocations = [GhostBlock.from_dict(ghostLoc) for ghostLoc in state.values()
                                        if 'is_drop_zone' in ghostLoc and 'GhostBlock' in ghostLoc['class_inheritance']
                                        and not ghostLoc['is_drop_zone']]

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                checkings = [room for room in self._roomLocations if not room.visited]
                if len(checkings) == 0:
                    self._phase = Phase.A
                    return None, {}
                #  pick a closed door
                # define the position of the door to be opened next
                next_door_position = -1;
                min_distance = 1000;
                # find the index of the closed door that is closest to the agent
                for i in range(len(self._roomLocations)):
                    if not self._roomLocations[i].visited:
                        x_difference = self._roomLocations[i].location[0] - state[agent_name]['location'][0]
                        y_difference = self._roomLocations[i].location[1] - state[agent_name]['location'][1]
                        distance = np.sqrt((x_difference ** 2) + (y_difference ** 2))
                        if (distance <= min_distance):
                            min_distance = distance
                            next_door_position = i
                self._roomLocations[next_door_position].visited = True
                self._door = self._roomLocations[next_door_position]
                # self._door = random.choice(closedDoors)
                doorLoc = self._door.location
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door.room_name, agent_name)
                #boolean to mark if the agent will quit its current task
                self._quit = round(random.uniform(0, 1), 0)

                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                if self._quit:
                    if round(random.uniform(0, 9), 0) < 3:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                        self._quit = False;
                        return None, {}

                self._state_tracker.update(state)
                # Follow path to door
                for msg in self.received_messages:
                    if self._door.room_name in msg.content:
                        if msg.from_id > self.agent_id:
                            self._quit = False;
                            self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                            return None, {}

                action = self._navigator.get_move_action(self._state_tracker)
                if action == None and self._quit:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                    return None, {}

                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                # Open door
                self._phase = Phase.ENTER_ROOM
                self._sendMessage('Opening door of ' + self._door.room_name, agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door.obj_id}

            if Phase.ENTER_ROOM == self._phase:
                self._navigator.reset_full()
                location = self._door.location
                location = location[0], location[1] - 1
                self._navigator.add_waypoints([location])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                #self._sendMessage('Entering room ' + self._door.room_name, agent_name)
                self._quit = round(random.uniform(0, 1), 0)
                if self._quit:
                    self._phase = Phase.ENTER_ROOM
                    self._quit = False
                    return None, {}
                if action != None:
                    return action, {}
                self._phase = Phase.SCAN_FOR_OBJECTS_2
                self._quit = round(random.uniform(0, 1), 0)
                return action, {}

            if Phase.D == self._phase:
                self._navigator.reset_full()
                self._phase = Phase.A
                ghostLoc = [GhostBlock.from_dict(ghostLoc) for ghostLoc in state.values()
                            if 'is_drop_zone' in ghostLoc and 'GhostBlock' in ghostLoc['class_inheritance']
                            and not ghostLoc['is_drop_zone']]

                temp = state[self.agent_id]['location'][0], \
                       state[self.agent_id]['location'][1] + 1
                check_myMessages = False
                check_otherMessages = False
                # Check if current agent previously dropped a goal block on the precedent location
                for msg in self._myMessages:
                    if str(temp) in msg.content:
                        check_myMessages = True
                        break
                # Check if other agents previously dropped a goal block on the precedent location
                for msg in self.received_messages:
                    if str(temp) in msg.content and "Dropped goal block" in msg.content:
                        check_otherMessages = True
                        break

                if check_myMessages or check_otherMessages or ghostLoc[0].loc == \
                            state[self.agent_id]['location']:
                    pickMeUp2 = state[self.agent_id]['is_carrying'][0]['visualization']
                    del pickMeUp2['depth']
                    del pickMeUp2['opacity']
                    del pickMeUp2['visualize_from_center']
                    self._sendMessage(
                            'Dropped goal block ' + str(pickMeUp2)
                            + ' at location ' + str(state[self.agent_id]['location']), self.agent_id)
                    self._myMessages.append(Message(content='Dropped goal block ' + str(pickMeUp2)
                            + ' at location ' + str(state[self.agent_id]['location']), from_id=self.agent_id))
                    return DropObject.__name__, {'object_id': state[self.agent_id]['is_carrying'][0]['obj_id']}

                self._phase = Phase.D
                return None, {}

            if Phase.C == self._phase:
                if len(state[self.agent_id]['is_carrying']) == 0:
                    self._phase = Phase.A
                    return None, {}

                self._navigator.reset_full()
                targetDrop = self._ghostLocations[0].loc
                self._navigator.add_waypoints([targetDrop])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}

                self._ghostLocations.remove(self._ghostLocations[0])
                self._phase = Phase.D

            if Phase.B == self._phase:
                targetPickUp = self._ghostLocations[0]
                self._navigator.reset_full()
                if state.get_of_type('CollectableBlock'):
                    pickMeUp = state.get_of_type('CollectableBlock')[0]
                    for availableBlock in state.get_of_type('CollectableBlock'):
                        # if availableBlock['visual']
                        temp = availableBlock['visualization']
                        if temp['shape'] == targetPickUp.shape and temp['colour'] == targetPickUp.colour:
                            pickMeUp = availableBlock
                            break
                    newPickMe = AvailableBlock.from_dict(pickMeUp['visualization'],pickMeUp['location'],
                                                         self.agent_id)
                    if targetPickUp.colour == newPickMe.colour and targetPickUp.shape == newPickMe.shape:
                        pickMeUp2 = pickMeUp['visualization']
                        del pickMeUp2['depth']
                        del pickMeUp2['opacity']
                        del pickMeUp2['visualize_from_center']
                        self._sendMessage('Picking up goal block ' + str(pickMeUp2)
                                          + ' at location ' + str(newPickMe.loc), self.agent_id)
                        self._phase = Phase.C
                        return GrabObject.__name__, {'object_id': pickMeUp['obj_id']}
                    else:
                        self._ghostLocations.remove(self._ghostLocations[0])
                        self._phase = Phase.A
                        return None, {}
                else:
                    self._ghostLocations.remove(self._ghostLocations[0])
                    self._phase = Phase.A
                    return None, {}

            if Phase.A == self._phase:
                self._navigator.reset_full()
                if len(self._ghostLocations) == 0:
                    self._phase = Phase.INIT
                    return None, {}
                targetPickUp = self._ghostLocations[0]
                checkings = [block for block in self._availableLocations if block.shape == targetPickUp.shape
                             and block.colour == targetPickUp.colour]

                if len(checkings) > 0:
                    current_target = checkings[0]
                    for msg in self.received_messages:
                        if str(current_target.loc) in msg.content and "Picking up goal block" in msg.content:
                            if msg.from_id > self.agent_id:
                                self._ghostLocations.remove(self._ghostLocations[0])
                                self._phase = Phase.A
                                return None, {}

                    self._navigator.add_waypoints([current_target.loc])
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        return action, {}
                    self._phase = Phase.B  # TODO grab object
                else:
                    for msg in self.received_messages:
                        if not "Dropped goal block" in msg.content:

                            self.received_messages.remove(msg)
                    self._roomLocations.clear()
                    temp = [door for door in state.values()
                            if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                    for door in temp:
                        if not door['location'] in self._roomLocations:
                            self._roomLocations.append(Room.from_dict(door))
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                    return None, {}

            if Phase.SCAN_FOR_OBJECTS_2 == self._phase:
                self._navigator.reset_full()
                if state.get_of_type('CollectableBlock'):
                    pickMeUp = state.get_of_type('CollectableBlock')[0]
                    visPickMe = pickMeUp['visualization']

                    current_block = AvailableBlock.from_dict(visPickMe,
                                                             pickMeUp['location'],
                                                             self.agent_id)

                    checkings = [block for block in self._availableLocations
                                 if block.loc == current_block.loc and block.shape == current_block.shape
                                 and block.colour == current_block.colour]
                    if current_block in self._ghostLocations and len(checkings) == 0:
                        self._quit = round(random.uniform(0, 1), 0)
                        if self._quit:
                            self._phase = Phase.ENTER_ROOM
                            self._quit = False
                            return None, {}
                        self._availableLocations.append(current_block)
                        pickMeUp2 = visPickMe
                        del pickMeUp2['depth']
                        del pickMeUp2['opacity']
                        del pickMeUp2['visualize_from_center']
                        self._sendMessage('Found goal block at ' + str(pickMeUp2)
                                          + ' at location ' +
                                          str(pickMeUp['location']), agent_name)

                location = self._door.location[0], self._door.location[1] - 1
                if state[self.agent_id]['location'] == location:
                    self._sendMessage('Searching through ' + self._door.room_name, agent_name)
                if self._flag:
                    if self.checkLeft(state):
                        target = state[self.agent_id]['location'][0] - 1, state[self.agent_id]['location'][1]

                    elif self.checkUp(state):
                        self._flag = not self._flag
                        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1

                    else:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                        return None, {}
                else:
                    if self.checkRight(state):
                        target = state[self.agent_id]['location'][0] + 1, state[self.agent_id]['location'][1]
                    elif self.checkUp(state):
                        self._flag = not self._flag
                        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1
                    else:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                        return None, {}

                self._navigator.add_waypoints([target])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if self._quit:
                    if round(random.uniform(0, 9), 0) < 3:
                        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                        self._quit = False;
                        return None, {}
                return action, {}

    def checkDirection(self, target, state):
        traverse_map = state.get_traverse_map()
        return traverse_map[target]

    def checkLeft(self, state):
        target = state[self.agent_id]['location'][0] - 1, state[self.agent_id]['location'][1]
        return self.checkDirection(target, state)

    def checkRight(self, state):
        target = state[self.agent_id]['location'][0] + 1, state[self.agent_id]['location'][1]
        return self.checkDirection(target, state)

    def checkUp(self, state):
        target = state[self.agent_id]['location'][0], state[self.agent_id]['location'][1] - 1
        return self.checkDirection(target, state)

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def _trustBlief(self, member, received):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # You can change the default value to your preference
        default = 0.5
        trustBeliefs = {}
        for member in received.keys():
            trustBeliefs[member] = default
        for member in received.keys():
            for message in received[member]:
                if 'Found' in message and 'colour' not in message:
                    trustBeliefs[member] -= 0.1
                    break
        return trustBeliefs
