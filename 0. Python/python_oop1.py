# Python Object-Oriented Programming

#### Class 기본 ####

class Employee:
    pass

emp1 = Employee()
emp2 = Employee()

# print(emp1)
# print(emp2)

emp1.first = 'Justin'
emp1.last = 'Sakong'
emp1.email = 'gjustin@naver.com'
emp1.pay = 50000

emp2.first = 'carol'
emp2.last = 'kim'
emp2.email = 'carol@naver.com'
emp2.pay = 60000


#### init 메소드 ####

class Employee:
    
    # init 메소드 추가
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = f'{first}_{last}@naver.com'
        
emp1 = Employee('Justin', 'Sakong', 50000)
emp2 = Employee('Carol', 'Kim', 60000)

# print(emp1.email)
# print(emp2.email)


#### Action 메소드 ####

class Employee:
    
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = f'{first}_{last}@naver.com'
    
    # 액션 추가
    def fullname(self):
        return f'{self.first} {self.last}'
                
emp1 = Employee('Justin', 'Sakong', 50000)
emp2 = Employee('Carol', 'Kim', 60000)

print(f'{emp1.first} {emp1.last}')
print(emp1.fullname())
