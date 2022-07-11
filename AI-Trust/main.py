from bw4t.BW4TWorld import BW4TWorld
from bw4t.statistics import Statistics
from agents1.BW4TBaselineAgent import BaseLineAgent, ColourblindAgent, StrongAgent, LazyAgent, LiarAgent
from agents1.BW4THuman import Human


"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [
        {'name':'agent13', 'botclass':BaseLineAgent, 'settings':{}},
        {'name':'agent2', 'botclass':StrongAgent, 'settings':{}},
        {'name': 'agent3', 'botclass': LiarAgent, 'settings': {}},
        {'name':'agent4', 'botclass': LazyAgent, 'settings':{}}
        ]

    print("Started world...")
    world=BW4TWorld(agents).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))
