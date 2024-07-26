BUILD_DIR=build
UNIT_TEST_DIR=test
UNIT_TEST_EXE=UnitTest

all:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR); cmake -DENABLE_UNIT_TEST=ON .. && $(MAKE)
	@cd $(BUILD_DIR)/$(UNIT_TEST_DIR); ./$(UNIT_TEST_EXE)

clean:
	@rm -rf $(BUILD_DIR)
