import unittest 
import parser

class ParserTest(unittest.TestCase):

    def test_get_passages(self):
        parser = parser.Parser()
        test_data = 'Lorem Ipsum is simply dummy text'

        parser.get_passages(test_data)


if __name__ == "__main__":
    unittest.main()