import math
import random

import pygame
import os
import config

from itertools import permutations
import copy

class BaseSprite(pygame.sprite.Sprite):
    images = dict()

    def __init__(self, x, y, file_name, transparent_color=None, wid=config.SPRITE_SIZE, hei=config.SPRITE_SIZE):
        pygame.sprite.Sprite.__init__(self)
        if file_name in BaseSprite.images:
            self.image = BaseSprite.images[file_name]
        else:
            self.image = pygame.image.load(os.path.join(config.IMG_FOLDER, file_name)).convert()
            self.image = pygame.transform.scale(self.image, (wid, hei))
            BaseSprite.images[file_name] = self.image
        # making the image transparent (if needed)
        if transparent_color:
            self.image.set_colorkey(transparent_color)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)


class Surface(BaseSprite):
    def __init__(self):
        super(Surface, self).__init__(0, 0, 'terrain.png', None, config.WIDTH, config.HEIGHT)


class Coin(BaseSprite):
    def __init__(self, x, y, ident):
        self.ident = ident
        super(Coin, self).__init__(x, y, 'coin.png', config.DARK_GREEN)

    def get_ident(self):
        return self.ident

    def position(self):
        return self.rect.x, self.rect.y

    def draw(self, screen):
        text = config.COIN_FONT.render(f'{self.ident}', True, config.BLACK)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)


class CollectedCoin(BaseSprite):
    def __init__(self, coin):
        self.ident = coin.ident
        super(CollectedCoin, self).__init__(coin.rect.x, coin.rect.y, 'collected_coin.png', config.DARK_GREEN)

    def draw(self, screen):
        text = config.COIN_FONT.render(f'{self.ident}', True, config.RED)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)


class Agent(BaseSprite):
    def __init__(self, x, y, file_name):
        super(Agent, self).__init__(x, y, file_name, config.DARK_GREEN)
        self.x = self.rect.x
        self.y = self.rect.y
        self.step = None
        self.travelling = False
        self.destinationX = 0
        self.destinationY = 0

    def set_destination(self, x, y):
        self.destinationX = x
        self.destinationY = y
        self.step = [self.destinationX - self.x, self.destinationY - self.y]
        magnitude = math.sqrt(self.step[0] ** 2 + self.step[1] ** 2)
        self.step[0] /= magnitude
        self.step[1] /= magnitude
        self.step[0] *= config.TRAVEL_SPEED
        self.step[1] *= config.TRAVEL_SPEED
        self.travelling = True

    def move_one_step(self):
        if not self.travelling:
            return
        self.x += self.step[0]
        self.y += self.step[1]
        self.rect.x = self.x
        self.rect.y = self.y
        if abs(self.x - self.destinationX) < abs(self.step[0]) and abs(self.y - self.destinationY) < abs(self.step[1]):
            self.rect.x = self.destinationX
            self.rect.y = self.destinationY
            self.x = self.destinationX
            self.y = self.destinationY
            self.travelling = False

    def is_travelling(self):
        return self.travelling

    def place_to(self, position):
        self.x = self.destinationX = self.rect.x = position[0]
        self.y = self.destinationX = self.rect.y = position[1]

    # coin_distance - cost matrix
    # return value - list of coin identifiers (containing 0 as first and last element, as well)
    def get_agent_path(self, coin_distance):
        pass


class ExampleAgent(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        path = [i for i in range(1, len(coin_distance))]
        print(coin_distance)
        random.shuffle(path)
        return [0] + path + [0]


class Aki(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        used=[]
        trenutnipolozaj=0
        cnt=0
        while(cnt<len(coin_distance)-1):
            minc = -1
            dalji = -1
            for i in range(1,len(coin_distance)):
                if i not in used:
                    if((minc==-1)or(coin_distance[trenutnipolozaj][i]<minc)):
                        minc=coin_distance[trenutnipolozaj][i]
                        dalji=i
            trenutnipolozaj=dalji
            used.append(trenutnipolozaj)
            cnt+=1

        return [0]+used+[0]


class Jocke(Agent):
    def __init__(self, x, y, file_name):
         super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        curr_path = [i for i in range(1, len(coin_distance))]
        maxc=-1
        listofp=list(permutations(curr_path))
        for i in listofp: #Iteracija kroz listu permutacija
            currc=0   #Trenutna cena tekuce permutacije
            tren = 0  # Trenutna lokacija na kojoj se nalazim, j je lokacija na kojus e prebacujem
            for j in i:
                currc+=coin_distance[tren][j] #ovo predstavlja cenu od trenutne do lokacije j
                tren=j #Prebacujem se na lokaciju j
            currc+=coin_distance[tren][0] #Nakon svih prebacivanja uzima se cena trenutne lokacije i lokacije j
            if((maxc==-1)or(currc<maxc)): #Ukoliko je prva cena ili cena manja od trenutne maksimalne, onda se dodaje nova konacna putanja
                maxc=currc
                curr_path=i
        return [0]+list(curr_path)+[0]


class Uki(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        parc=[[0]]
        cene=[0]
        while(1):
            minimc=min(cene)
            getnext=[]

            for i in range(0, len(cene)):
                if(cene[i]==minimc):
                    getnext.append(i)
            inde=getnext[0]
            #print(parc)
            if(len(getnext)>1):
                #print("Usao u prvu petlju:")
                #print(getnext)
                maxd=0
                duzine=[]
                for i in getnext:
                    if(len(parc[i])>maxd):
                        duzine.clear()
                        duzine.append(i)
                        maxd=len(parc[i])
                    elif(len(parc[i])==maxd):
                        duzine.append(i)
                        #print("tu sam")
                inde=duzine[0]
                #print("duzine, maxd, inde:")
                #print(duzine)
                #print(maxd)
                #print(inde)
                if(len(duzine)>1):
                    #print("usao")
                    for i in range(0, len(duzine)-1):
                        #print("Parc[i+1]:")
                        #print(parc[i+1])
                        if(parc[duzine[i+1]][maxd-1]<parc[duzine[i]][maxd-1]):
                            inde=duzine[i+1]
                   #print("izasao")
            #print(inde)
            tren=parc.pop(inde)
            cena=cene.pop(inde)
            ind=tren[len(tren)-1]
            #print("Tren, cena, ind:")
            #print(tren)
            #print(cena)
            #print(ind)
            if (len(tren) == len(coin_distance)):
                cene.append(cena + coin_distance[ind][0])
                pom = copy.deepcopy(tren)
                parc.append(pom+[0])
            elif(len(tren) == len(coin_distance)+1):
                break
            else:
                cnt=0
                for i in coin_distance[ind]:
                    if cnt not in tren:
                        cene.append(cena+i)
                        pom=copy.deepcopy(tren)
                        parc.append(pom+[cnt])
                    cnt+=1

        return tren


class Grana:
    def __init__(self, n1, n2, cena):
        self.n1 = n1
        self.n2 = n2
        self.cena = cena


class Micko(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        parc = [[0]]
        cene = [0]
        cene_bez_heuristike = [0]
        while (1):
            minimc = min(cene)
            getnext = []

            for i in range(0, len(cene)):
                if (cene[i] == minimc):
                    getnext.append(i)
            inde = getnext[0]
            # print(parc)
            if (len(getnext) > 1):
                # print("Usao u prvu petlju:")
                # print(getnext)
                maxd = 0
                duzine = []
                for i in getnext:
                    if (len(parc[i]) > maxd):
                        duzine.clear()
                        duzine.append(i)
                        maxd = len(parc[i])
                    elif (len(parc[i]) == maxd):
                        duzine.append(i)
                        # print("tu sam")
                inde = duzine[0]
                # print("duzine, maxd, inde:")
                # print(duzine)
                # print(maxd)
                # print(inde)
                if (len(duzine) > 1):
                    # print("usao")
                    for i in range(0, len(duzine) - 1):
                        # print("Parc[i+1]:")
                        # print(parc[i+1])
                        if (parc[duzine[i + 1]][maxd - 1] < parc[duzine[i]][maxd - 1]):
                            inde = duzine[i + 1]
                # print("izasao")
            # print(inde)
            tren = parc.pop(inde)
            cene.pop(inde)
            cena = cene_bez_heuristike.pop(inde)
            ind = tren[len(tren) - 1]
            # print("Tren, cena, ind:")
            # print(tren)
            # print(cena)
            # print(ind)
            if (len(tren) == len(coin_distance)):
                cene.append(cena + coin_distance[ind][0])
                cene_bez_heuristike.append(cena + coin_distance[ind][0])
                pom = copy.deepcopy(tren)
                parc.append(pom + [0])
            elif (len(tren) == len(coin_distance) + 1):
                break
            else:
                cnt = 0
                for i in coin_distance[ind]:
                    if cnt not in tren:
                        #cene.append(cena + i)
                        mos=[]
                        grane=[]
                        for j in range(1, len(tren)):
                            mos.append(tren[j])
                        for j in range(0, len(coin_distance)):
                            for k in range(j+1, len(coin_distance)):
                                if j not in mos and k not in mos:
                                    g=Grana(j, k, coin_distance[j][k])
                                    grane.append(g)
                        #nakon ovoga ide sortiranje grana
                        grane=(sorted(grane, key=lambda x: x.cena))
                        #print(grane)
                        iskoriscene=[]
                        ccena=0
                        #Treba ovde brojac da se doda dok se ne izadje iz petlje
                        brojac=0
                        for j in grane:
                           #print(iskoriscene)
                            dali = 1
                            if(brojac==len(coin_distance)-len(mos)-1):
                                break
                            for k in iskoriscene:
                                if j.n1 in k and j.n2 in k:
                                    dali=0
                            if(dali==1):
                                brojac+=1
                                dali2=0
                                gen=-1
                                for k in range(0, len(iskoriscene)):
                                    if(k<len(iskoriscene)): #Ovo je ispravka
                                        if j.n1 in iskoriscene[k]:
                                            if(dali2==0):
                                                iskoriscene[k].append(j.n2)
                                                dali2=1
                                                gen=k
                                            elif(dali2==1):
                                                dali2=2
                                                tempr=iskoriscene.pop(gen)
                                                for l in tempr:
                                                    if(l!=j.n1):
                                                        iskoriscene[gen].append(l)
                                        elif j.n2 in iskoriscene[k]:
                                            if (dali2 == 0):
                                                iskoriscene[k].append(j.n1)
                                                dali2=1
                                                gen=k
                                            elif (dali2 == 1):
                                                dali2=2
                                                tempr = iskoriscene.pop(gen)
                                                for l in tempr:
                                                    if (l != j.n2):
                                                        iskoriscene[gen].append(l)
                                if(dali2==0):
                                    iskoriscene.append([j.n1, j.n2])
                                    #print(iskoriscene)
                                ccena += j.cena
                        cene.append(cena+ccena+i)
                        cene_bez_heuristike.append(cena + i)
                        pom = copy.deepcopy(tren)
                        parc.append(pom + [cnt])
                    cnt += 1

        return tren