class Route:
    def __init__(self):
        self.route = []
        self.load = 0

    def add_node(self, node):
        self.route.append(node)

    def remove_node(self, node):
        if node in self.route:
            self.route.remove(node)

    def get_route(self):
        return self.route

    def get_load(self):
        return self.load

    def update_load(self, demand):
        self.load += demand

    def is_empty(self):
        return len(self.route) == 0

    def get_last_node(self):
        return self.route[-1] if not self.is_empty() else None
