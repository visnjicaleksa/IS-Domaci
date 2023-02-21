import pygame
import os
import config
import copy
import math

class BaseSprite(pygame.sprite.Sprite):
    images = dict()

    def __init__(self, row, col, file_name, transparent_color=None):
        pygame.sprite.Sprite.__init__(self)
        if file_name in BaseSprite.images:
            self.image = BaseSprite.images[file_name]
        else:
            self.image = pygame.image.load(os.path.join(config.IMG_FOLDER, file_name)).convert()
            self.image = pygame.transform.scale(self.image, (config.TILE_SIZE, config.TILE_SIZE))
            BaseSprite.images[file_name] = self.image
        # making the image transparent (if needed)
        if transparent_color:
            self.image.set_colorkey(transparent_color)
        self.rect = self.image.get_rect()
        self.rect.topleft = (col * config.TILE_SIZE, row * config.TILE_SIZE)
        self.row = row
        self.col = col


class Agent(BaseSprite):
    def __init__(self, row, col, file_name):
        super(Agent, self).__init__(row, col, file_name, config.DARK_GREEN)

    def move_towards(self, row, col):
        row = row - self.row
        col = col - self.col
        self.rect.x += col
        self.rect.y += row

    def place_to(self, row, col):
        self.row = row
        self.col = col
        self.rect.x = col * config.TILE_SIZE
        self.rect.y = row * config.TILE_SIZE

    # game_map - list of lists of elements of type Tile
    # goal - (row, col)
    # return value - list of elements of type Tile
    def get_agent_path(self, game_map, goal):
        pass


class ExampleAgent(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = [game_map[self.row][self.col]]

        row = self.row
        col = self.col
        while True:
            if row != goal[0]:
                row = row + 1 if row < goal[0] else row - 1
            elif col != goal[1]:
                col = col + 1 if col < goal[1] else col - 1
            else:
                break
            path.append(game_map[row][col])
        return path



class Aki(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = [game_map[self.row][self.col]]

        row = self.row
        col = self.col
        stack=[[self.row, self.col]]
        putanja=[]
        stack_temp=[]
        while(1):
            for x in range(row-1, row+2):
                for y in range(col-1, col+2):
                    if((y==col)and(x!=row))or((y!=col)and(x==row)):
                        if((x>=0)and(x<len(game_map))and((y>=0)and(y<len(game_map[0])))):
                            n=[x,y]
                            if(putanja.count(n)==0):
                                stack_temp.append([x,y])
            for i in range(0, len(stack_temp)):
                for j in range(i+1,len(stack_temp)):
                    if(game_map[stack_temp[j][0]][stack_temp[j][1]].cost()<game_map[stack_temp[i][0]][stack_temp[i][1]].cost()):
                        stack_temp[j], stack_temp[i]=stack_temp[i], stack_temp[j]
                    elif (game_map[stack_temp[j][0]][stack_temp[j][1]].cost()==game_map[stack_temp[i][0]][stack_temp[i][1]].cost()):
                        if(stack_temp[j][0]==row-1):
                            stack_temp[j], stack_temp[i] = stack_temp[i], stack_temp[j]
                        elif(((stack_temp[j][1]==col+1))and(stack_temp[i][0]!=row-1)):
                            stack_temp[j], stack_temp[i] = stack_temp[i], stack_temp[j]
                        elif((stack_temp[j][0]==row+1)and(stack_temp[i][1]==col-1)):
                            stack_temp[j], stack_temp[i] = stack_temp[i], stack_temp[j]
            for i in reversed(stack_temp):
                stack.append(i);
            v=stack.pop();
            row=v[0]
            col=v[1]
            putanja.append([row,col])
            stack_temp.clear()
            if (row==goal[0] and col==goal[1]):
                break;
        for l in putanja:
            path.append(game_map[l[0]][l[1]])
        return path

class Jocke(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = [game_map[self.row][self.col]]

        row = self.row
        col = self.col
        trenutni=[[row, col]]
        queue=[]
        queue2=[]
        stack_temp=[]
        while(1):
            for x in range(row-1, row+2):
                for y in range(col-1, col+2):
                    if((y==col)and(x!=row))or((y!=col)and(x==row)):
                        if((x>=0)and(x<len(game_map))and((y>=0)and(y<len(game_map[0])))):
                            n=[x,y]
                            if(trenutni.count(n)==0):
                                stack_temp.append([x,y])
            for i in range(0, len(stack_temp)):
                for j in range(i+1,len(stack_temp)):
                    m=stack_temp[i]
                    n=stack_temp[j]
                    summ = 0
                    brm=0
                    sumn = 0
                    brn=0
                    for x in range(m[0] - 1, m[0] + 2):
                        for y in range(m[1] - 1, m[1] + 2):
                            if ((y == m[1]) and (x != m[0])) or ((y != m[1]) and (x == m[0])):
                                if ((x >= 0) and (x < len(game_map)) and ((y >= 0) and (y < len(game_map[0])))):
                                    if((row!=x)or(col!=y)):
                                        summ+=game_map[x][y].cost()
                                        brm+=1
                    summ/=brm
                    for x in range(n[0] - 1, n[0] + 2):
                        for y in range(n[1] - 1, n[1] + 2):
                            if ((y == n[1]) and (x != n[0])) or ((y != n[1]) and (x == n[0])):
                                if ((x >= 0) and (x < len(game_map)) and ((y >= 0) and (y < len(game_map[0])))):
                                    if ((row != x) or (col != y)):
                                        sumn+=game_map[x][y].cost()
                                        brn+=1
                    summ /= brm
                    if(sumn<summ):
                        stack_temp[j], stack_temp[i]=stack_temp[i], stack_temp[j]
                    elif (sumn==summ):
                        if(stack_temp[j][0]==row-1):
                            stack_temp[j], stack_temp[i] = stack_temp[i], stack_temp[j]
                        elif(((stack_temp[j][1]==col+1))and(stack_temp[i][0]!=row-1)):
                            stack_temp[j], stack_temp[i] = stack_temp[i], stack_temp[j]
                        elif((stack_temp[j][0]==row+1)and(stack_temp[i][1]==col-1)):
                            stack_temp[j], stack_temp[i] = stack_temp[i], stack_temp[j]
            for i in stack_temp:
                trenutni2=copy.deepcopy(trenutni)
                trenutni2.append(i)
                queue.append(copy.deepcopy(trenutni2))
                queue2.append(copy.deepcopy(trenutni2))
                trenutni2.pop()

            trenutni=queue2.pop(0)
            row=trenutni[len(trenutni)-1][0]
            col=trenutni[len(trenutni)-1][1]
            stack_temp.clear()
            if (row==goal[0] and col==goal[1]):
                break;
        for l in trenutni:
            if((l[0]!=self.row)or(l[1]!=self.col)):
                path.append(game_map[l[0]][l[1]])
        return path

class Draza(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = [game_map[self.row][self.col]]
        min_cena=-1
        row = self.row
        col = self.col
        trenutni=[[row, col]]
        najbrzi=[]
        queue=[]
        queue2=[]
        stack_temp=[]
        brisac=[]
        while(1):
            for x in range(row-1, row+2):
                for y in range(col-1, col+2):
                    if((y==col)and(x!=row))or((y!=col)and(x==row)):
                        if((x>=0)and(x<len(game_map))and((y>=0)and(y<len(game_map[0])))):
                            n=[x,y]
                            if(trenutni.count(n)==0):
                                stack_temp.append([x,y])
            if(len(stack_temp)>0):
                for i in stack_temp:
                    trenutni2=copy.deepcopy(trenutni)
                    trenutni2.append(i)
                    if((i[0]==goal[0])and(i[1]==goal[1])):
                        cena = 0
                        for e in range(1, len(trenutni2)):
                            cena += game_map[trenutni2[e][0]][trenutni2[e][1]].cost()
                        if((min_cena==-1)or(cena<min_cena)):
                            min_cena=cena
                            najbrzi=copy.deepcopy(trenutni2)

                    else:
                        queue.append(copy.deepcopy(trenutni2))
                    trenutni2.pop()

            mcena=-1
            k=-1
            for it in range(0, len(queue)):
                cena = 0
                for jo in range(1, len(queue[it])):
                    cena += game_map[queue[it][jo][0]][queue[it][jo][1]].cost()
                if ((mcena == -1) or (cena < mcena)):
                    mcena = cena
                    k=it
                elif((cena==mcena)and(len(queue[it])<len(queue[k]))):
                    mcena = cena
                    k = it
                if ((min_cena != -1) and (cena > min_cena)):
                    brisac.append(copy.copy(queue[it]))
            trenutni=queue.pop(k)
            if(len(brisac)>0):
               for ka in brisac:
                   queue.remove(ka)
            brisac.clear()
            row=trenutni[len(trenutni)-1][0]
            col=trenutni[len(trenutni)-1][1]
            stack_temp.clear()
            if (len(queue)==0):
                break;
        for l in najbrzi:
            if((l[0]!=self.row)or(l[1]!=self.col)):
                path.append(game_map[l[0]][l[1]])
        return path

class Bole(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = [game_map[self.row][self.col]]
        min_cena=-1
        row = self.row
        col = self.col
        trenutni=[[row, col]]
        najbrzi=[]
        queue=[]
        queue2=[]
        stack_temp=[]
        brisac=[]
        while(1):
            for x in range(row-1, row+2):
                for y in range(col-1, col+2):
                    if((y==col)and(x!=row))or((y!=col)and(x==row)):
                        if((x>=0)and(x<len(game_map))and((y>=0)and(y<len(game_map[0])))):
                            n=[x,y]
                            if(trenutni.count(n)==0):
                                stack_temp.append([x,y])
            if(len(stack_temp)>0):
                for i in stack_temp:
                    trenutni2=copy.deepcopy(trenutni)
                    trenutni2.append(i)
                    if((i[0]==goal[0])and(i[1]==goal[1])):
                        cena = 0
                        for e in range(1, len(trenutni2)):
                            cena += game_map[trenutni2[e][0]][trenutni2[e][1]].cost()
                        if((min_cena==-1)or(cena<min_cena)):
                            min_cena=cena
                            najbrzi=copy.deepcopy(trenutni2)

                    else:
                        queue.append(copy.deepcopy(trenutni2))
                    trenutni2.pop()

            mcena=-1
            k=-1
            for it in range(0, len(queue)):
                cena = 0
                for jo in range(1, len(queue[it])):
                    cena += game_map[queue[it][jo][0]][queue[it][jo][1]].cost()
                cena+=math.sqrt((queue[it][len(queue[it])-1][0]-goal[0])**2+(queue[it][len(queue[it])-1][1]-goal[1])**2)
                if ((mcena == -1) or (cena < mcena)):
                    mcena = cena
                    k=it
                elif((cena==mcena)and(len(queue[it])<len(queue[k]))):
                    mcena = cena
                    k = it
                if ((min_cena != -1) and (cena > min_cena)):
                    brisac.append(copy.copy(queue[it]))
            trenutni=queue.pop(k)
            if(len(brisac)>0):
               for ka in brisac:
                   queue.remove(ka)
            brisac.clear()
            row=trenutni[len(trenutni)-1][0]
            col=trenutni[len(trenutni)-1][1]
            stack_temp.clear()
            if (len(queue)==0):
                break;
        for l in najbrzi:
            if((l[0]!=self.row)or(l[1]!=self.col)):
                path.append(game_map[l[0]][l[1]])
        return path

class Tile(BaseSprite):
    def __init__(self, row, col, file_name):
        super(Tile, self).__init__(row, col, file_name)

    def position(self):
        return self.row, self.col

    def cost(self):
        pass

    def kind(self):
        pass


class Stone(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'stone.png')

    def cost(self):
        return 1000

    def kind(self):
        return 's'


class Water(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'water.png')

    def cost(self):
        return 500

    def kind(self):
        return 'w'


class Road(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'road.png')

    def cost(self):
        return 2

    def kind(self):
        return 'r'


class Grass(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'grass.png')

    def cost(self):
        return 3

    def kind(self):
        return 'g'


class Mud(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'mud.png')

    def cost(self):
        return 5

    def kind(self):
        return 'm'


class Dune(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'dune.png')

    def cost(self):
        return 7

    def kind(self):
        return 's'


class Goal(BaseSprite):
    def __init__(self, row, col):
        super().__init__(row, col, 'x.png', config.DARK_GREEN)


class Trail(BaseSprite):
    def __init__(self, row, col, num):
        super().__init__(row, col, 'trail.png', config.DARK_GREEN)
        self.num = num

    def draw(self, screen):
        text = config.GAME_FONT.render(f'{self.num}', True, config.WHITE)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)
