from unittest import TestCase, main


class SampleTest(TestCase):

    def test_trivial(self):
        self.assertEquals(1 + 1, 2)


if __name__ == "__main__":
    main()
