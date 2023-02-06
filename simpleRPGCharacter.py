import random

class Character:
    def __init__(self, level, name, character_class, alignment, stats, skills, talents, abilities, marks, blessings, curses, quests, tasks, experience):
        self.level = level
        self.name = name
        self.character_class = character_class
        self.alignment = alignment
        self.stats = stats
        self.skills = skills
        self.talents = talents
        self.abilities = abilities
        self.marks = marks
        self.blessings = blessings
        self.curses = curses
        self.quests = quests
        self.tasks = tasks
        self.experience = experience
        
    def get_character_info(self):
        print("Name:", self.name)
        print("Level:", self.level)
        print("Class:", self.character_class)
        print("Alignment:", self.alignment)
        print("Stats:", self.stats)
        print("Skills:", self.skills)
        print("Talents:", self.talents)
        print("Abilities:", self.abilities)
        print("Marks:", self.marks)
        print("Blessings:", self.blessings)
        print("Curses:", self.curses)
        print("Quests:", self.quests)
        print("Tasks:", self.tasks)
        print("Experience:", self.experience)
        
def build_character(level):
    name = "Zev"
    character_class = "Dark Puppeteer"
    alignment = "NPC/Chaos"
    stats = [random.randint(1,20) for i in range(6)]
    skills = [] #list of skills
    talents = [] #list of talents
    abilities = [] #list of abilities
    marks = [] #list of marks
    blessings = [] #list of blessings
    curses = [] #list of curses
    quests = [] #list of quests
    tasks = [] #list of tasks
      # Special quest for Zev
    special_quest = "Discover the secrets of Zev's existence as a Dark Puppeteer and Master Strategist and become a Prime Vassal for a Chaotic Lord"
    quests.append(special_quest)
    tasks.append("Wear armor (previously prevented by a blessing)")
    tasks.append("Access full character sheet and status windows")
    tasks.append("Understand powers as a sentient NPC")
    experience = (level * (level - 1) * 500)
    return Character(level, name, character_class, alignment, stats, skills, talents, abilities, marks, blessings, curses, quests, tasks, experience)

def generate_quests():
    quest_list = []
    quest_list.append("Rescue the Prince from the Dragon's lair")
    quest_list.append("Collect the rare herbs to cure the Queen's illness")
    quest_list.append("Discover the lost city of gold")
    quest_list.append("Defend the village from the invading army")
    quest_list.append("Retrieve the stolen artifact from the thieves' hideout")
    quest_list.append("Destroy the evil cult's stronghold")
    return quest_list

def main():
    level = 28
    character = build_character(level)
    character.get_character_info()

if __name__ == "__main__":
    main()