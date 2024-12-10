import radiation

search_area_x = 50
search_area_y = 50

agent_x = 0
agent_y = 0
agent_moveDist = 1
agent = radiation.agent(agent_x, agent_y, search_area_x, search_area_y, agent_moveDist)
agent_positions = [] # Save past agent positions


actions = [agent.moveUp, agent.moveDown, agent.moveLeft, agent.moveRight]

print(agent.state())