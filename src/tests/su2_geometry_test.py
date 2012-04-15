

class testHopf(unittest.TestCase):

    def setUp(self):
        self.N = 17 # Number of rows to create -- esssentially arbitrary

        # Random complex matrix
        self.psi = np.random.rand(self.N, 2) + \
                1j*np.random.rand(self.N, 2)




    def testPhase(self):
        
        X = su2_geometry.hopf(self.psi)

        theta = np.random.rand(self.N, 1) # Column vector of phases






if __name__ == '__main__':
    unittest.main()

