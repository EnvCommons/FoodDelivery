from openreward.environments import Server

from fooddelivery import FoodDelivery

if __name__ == "__main__":
    server = Server([FoodDelivery])
    server.run()
