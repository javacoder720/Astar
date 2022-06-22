from driver import Driver

driver = Driver()

driver.generate_maps(1,1)
driver.save_maps()
driver.show_maps()
driver.show_maps_and_pairs()

driver.load_maps("test_maps","test_pairs")
driver.run_agents(show=True)
input()
driver.run_sequential_agents(show=True)
input()

driver.load_maps()
driver.run_agents()
input()
driver.run_sequential_agents()
