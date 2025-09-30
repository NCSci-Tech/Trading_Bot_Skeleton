// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

// Interface for DEX routers (Uniswap V2 style)
interface IDEXRouter {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    
    function getAmountsOut(uint amountIn, address[] calldata path)
        external view returns (uint[] memory amounts);
    
    function WETH() external pure returns (address);
}

// Interface for price oracles
interface IPriceOracle {
    function getPrice(address token) external view returns (uint256);
    function getLatestRoundData(address token) external view returns (
        uint80 roundId,
        int256 price,
        uint256 startedAt,
        uint256 updatedAt,
        uint80 answeredInRound
    );
}

/**
 * @title ArbitrageBot
 * @dev Smart contract for executing arbitrage trades across multiple DEXs
 */
contract ArbitrageBot is Ownable, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;
    
    struct ArbitrageParams {
        address tokenA;
        address tokenB;
        address dexA;
        address dexB;
        uint256 minProfitBps; // Minimum profit in basis points
        uint256 maxSlippageBps; // Maximum slippage in basis points
        uint256 maxGasPrice; // Maximum gas price for execution
    }
    
    struct ArbitrageResult {
        uint256 amountIn;
        uint256 amountOut;
        uint256 profit;
        uint256 gasUsed;
        uint256 timestamp;
        bool successful;
    }
    
    struct DEXInfo {
        address router;
        uint256 fee; // Fee in basis points
        bool active;
        string name;
    }
    
    // Constants
    uint256 public constant BASIS_POINTS = 10000;
    uint256 public constant MAX_SLIPPAGE = 1000; // 10%
    uint256 public constant MIN_PROFIT_BPS = 10; // 0.1%
    
    // State variables
    mapping(address => bool) public authorizedCallers;
    mapping(address => mapping(address => address)) public tokenPairs; // tokenA -> tokenB -> pair address
    mapping(address => DEXInfo) public dexInfo;
    mapping(bytes32 => ArbitrageResult) public arbitrageHistory;
    
    address[] public supportedDEXs;
    address public priceOracle;
    address public treasury;
    
    uint256 public totalProfitGenerated;
    uint256 public totalArbitrageExecuted;
    uint256 public maxGasPrice = 50 gwei;
    uint256 public profitShareBps = 500; // 5% profit share to treasury
    
    // Events
    event ArbitrageExecuted(
        bytes32 indexed tradeId,
        address indexed tokenA,
        address indexed tokenB,
        uint256 amountIn,
        uint256 profit,
        address dexBought,
        address dexSold
    );
    
    event DEXAdded(address indexed dex, string name, uint256 fee);
    event DEXUpdated(address indexed dex, bool active);
    event ProfitWithdrawn(address indexed recipient, uint256 amount);
    event MaxGasPriceUpdated(uint256 oldPrice, uint256 newPrice);
    event AuthorizedCallerUpdated(address indexed caller, bool authorized);
    
    // Modifiers
    modifier onlyAuthorized() {
        require(authorizedCallers[msg.sender] || msg.sender == owner(), "ArbitrageBot: Not authorized");
        _;
    }
    
    modifier gasThrottle() {
        require(tx.gasprice <= maxGasPrice, "ArbitrageBot: Gas price too high");
        _;
    }
    
    modifier validToken(address token) {
        require(token != address(0), "ArbitrageBot: Invalid token address");
        _;
    }
    
    constructor(address _priceOracle, address _treasury) {
        require(_priceOracle != address(0), "ArbitrageBot: Invalid oracle");
        require(_treasury != address(0), "ArbitrageBot: Invalid treasury");
        
        priceOracle = _priceOracle;
        treasury = _treasury;
        authorizedCallers[msg.sender] = true;
    }
    
    /**
     * @dev Add authorized caller
     */
    function addAuthorizedCaller(address caller) external onlyOwner {
        require(caller != address(0), "ArbitrageBot: Invalid caller");
        authorizedCallers[caller] = true;
        emit AuthorizedCallerUpdated(caller, true);
    }
    
    /**
     * @dev Remove authorized caller
     */
    function removeAuthorizedCaller(address caller) external onlyOwner {
        authorizedCallers[caller] = false;
        emit AuthorizedCallerUpdated(caller, false);
    }
    
    /**
     * @dev Add a new DEX for arbitrage
     */
    function addDEX(
        address dex,
        address router,
        string memory name,
        uint256 fee
    ) external onlyOwner {
        require(dex != address(0) && router != address(0), "ArbitrageBot: Invalid addresses");
        require(bytes(name).length > 0, "ArbitrageBot: Invalid name");
        require(fee <= 1000, "ArbitrageBot: Fee too high"); // Max 10%
        
        dexInfo[dex] = DEXInfo({
            router: router,
            fee: fee,
            active: true,
            name: name
        });
        
        supportedDEXs.push(dex);
        emit DEXAdded(dex, name, fee);
    }
    
    /**
     * @dev Update DEX status
     */
    function updateDEXStatus(address dex, bool active) external onlyOwner {
        require(dexInfo[dex].router != address(0), "ArbitrageBot: DEX not found");
        dexInfo[dex].active = active;
        emit DEXUpdated(dex, active);
    }
    
    /**
     * @dev Check arbitrage opportunity between two DEXs
     */
    function checkArbitrageOpportunity(
        ArbitrageParams memory params,
        uint256 amountIn
    ) external view returns (bool profitable, uint256 expectedProfit, uint256 priceImpact) {
        require(dexInfo[params.dexA].active && dexInfo[params.dexB].active, "ArbitrageBot: Inactive DEX");
        
        // Get prices from both DEXs
        address[] memory pathAtoB = new address[](2);
        pathAtoB[0] = params.tokenA;
        pathAtoB[1] = params.tokenB;
        
        address[] memory pathBtoA = new address[](2);
        pathBtoA[0] = params.tokenB;
        pathBtoA[1] = params.tokenA;
        
        try IDEXRouter(dexInfo[params.dexA].router).getAmountsOut(amountIn, pathAtoB) returns (uint[] memory amountsA) {
            try IDEXRouter(dexInfo[params.dexB].router).getAmountsOut(amountsA[1], pathBtoA) returns (uint[] memory amountsB) {
                uint256 finalAmount = amountsB[1];
                
                if (finalAmount > amountIn) {
                    expectedProfit = finalAmount - amountIn;
                    uint256 profitBps = (expectedProfit * BASIS_POINTS) / amountIn;
                    profitable = profitBps >= params.minProfitBps;
                    
                    // Calculate price impact
                    priceImpact = ((amountIn - finalAmount) * BASIS_POINTS) / amountIn;
                } else {
                    profitable = false;
                    expectedProfit = 0;
                    priceImpact = ((amountIn - finalAmount) * BASIS_POINTS) / amountIn;
                }
            } catch {
                profitable = false;
                expectedProfit = 0;
                priceImpact = 0;
            }
        } catch {
            profitable = false;
            expectedProfit = 0;
            priceImpact = 0;
        }
    }
    
    /**
     * @dev Execute arbitrage trade
     */
    function executeArbitrage(
        ArbitrageParams memory params,
        uint256 amountIn,
        uint256 minProfitAmount,
        uint256 deadline
    ) external onlyAuthorized gasThrottle whenNotPaused nonReentrant {
        require(amountIn > 0, "ArbitrageBot: Invalid amount");
        require(deadline > block.timestamp, "ArbitrageBot: Expired deadline");
        require(params.minProfitBps >= MIN_PROFIT_BPS, "ArbitrageBot: Profit threshold too low");
        require(params.maxSlippageBps <= MAX_SLIPPAGE, "ArbitrageBot: Slippage too high");
        
        uint256 gasStart = gasleft();
        bytes32 tradeId = keccak256(abi.encodePacked(block.timestamp, msg.sender, amountIn));
        
        // Check opportunity is still profitable
        (bool profitable, uint256 expectedProfit,) = this.checkArbitrageOpportunity(params, amountIn);
        require(profitable && expectedProfit >= minProfitAmount, "ArbitrageBot: Not profitable");
        
        // Transfer tokens from caller
        IERC20(params.tokenA).safeTransferFrom(msg.sender, address(this), amountIn);
        
        uint256 initialBalance = IERC20(params.tokenA).balanceOf(address(this));
        
        try this._executeArbitrageTrade(params, amountIn, deadline) returns (uint256 profit) {
            uint256 gasUsed = gasStart - gasleft();
            
            // Record successful arbitrage
            arbitrageHistory[tradeId] = ArbitrageResult({
                amountIn: amountIn,
                amountOut: initialBalance + profit,
                profit: profit,
                gasUsed: gasUsed,
                timestamp: block.timestamp,
                successful: true
            });
            
            totalProfitGenerated += profit;
            totalArbitrageExecuted++;
            
            // Take profit share for treasury
            uint256 treasuryShare = (profit * profitShareBps) / BASIS_POINTS;
            if (treasuryShare > 0) {
                IERC20(params.tokenA).safeTransfer(treasury, treasuryShare);
                profit -= treasuryShare;
            }
            
            // Return remaining tokens to caller
            IERC20(params.tokenA).safeTransfer(msg.sender, amountIn + profit);
            
            emit ArbitrageExecuted(
                tradeId,
                params.tokenA,
                params.tokenB,
                amountIn,
                profit,
                params.dexA,
                params.dexB
            );
        } catch Error(string memory reason) {
            // Arbitrage failed, return tokens
            IERC20(params.tokenA).safeTransfer(msg.sender, amountIn);
            
            arbitrageHistory[tradeId] = ArbitrageResult({
                amountIn: amountIn,
                amountOut: 0,
                profit: 0,
                gasUsed: gasStart - gasleft(),
                timestamp: block.timestamp,
                successful: false
            });
            
            revert(string(abi.encodePacked("ArbitrageBot: Execution failed - ", reason)));
        }
    }
    
    /**
     * @dev Internal function to execute the actual arbitrage trade
     */
    function _executeArbitrageTrade(
        ArbitrageParams memory params,
        uint256 amountIn,
        uint256 deadline
    ) external returns (uint256 profit) {
        require(msg.sender == address(this), "ArbitrageBot: Internal function");
        
        // Step 1: Swap tokenA for tokenB on DEX A
        address[] memory pathAtoB = new address[](2);
        pathAtoB[0] = params.tokenA;
        pathAtoB[1] = params.tokenB;
        
        IERC20(params.tokenA).approve(dexInfo[params.dexA].router, amountIn);
        
        uint256[] memory amountsA = IDEXRouter(dexInfo[params.dexA].router).swapExactTokensForTokens(
            amountIn,
            0, // Accept any amount of tokenB
            pathAtoB,
            address(this),
            deadline
        );
        
        uint256 tokenBAmount = amountsA[1];
        
        // Step 2: Swap tokenB back to tokenA on DEX B
        address[] memory pathBtoA = new address[](2);
        pathBtoA[0] = params.tokenB;
        pathBtoA[1] = params.tokenA;
        
        IERC20(params.tokenB).approve(dexInfo[params.dexB].router, tokenBAmount);
        
        uint256[] memory amountsB = IDEXRouter(dexInfo[params.dexB].router).swapExactTokensForTokens(
            tokenBAmount,
            0, // Accept any amount of tokenA
            pathBtoA,
            address(this),
            deadline
        );
        
        uint256 finalAmount = amountsB[1];
        
        require(finalAmount > amountIn, "ArbitrageBot: No profit generated");
        
        profit = finalAmount - amountIn;
    }
    
    /**
     * @dev Batch check multiple arbitrage opportunities
     */
    function batchCheckOpportunities(
        ArbitrageParams[] memory paramsList,
        uint256[] memory amounts
    ) external view returns (
        bool[] memory profitable,
        uint256[] memory expectedProfits,
        uint256[] memory priceImpacts
    ) {
        require(paramsList.length == amounts.length, "ArbitrageBot: Array length mismatch");
        
        profitable = new bool[](paramsList.length);
        expectedProfits = new uint256[](paramsList.length);
        priceImpacts = new uint256[](paramsList.length);
        
        for (uint256 i = 0; i < paramsList.length; i++) {
            (profitable[i], expectedProfits[i], priceImpacts[i]) = 
                this.checkArbitrageOpportunity(paramsList[i], amounts[i]);
        }
    }
    
    /**
     * @dev Get price from oracle with staleness check
     */
    function getOraclePrice(address token) public view returns (uint256 price, bool isStale) {
        try IPriceOracle(priceOracle).getLatestRoundData(token) returns (
            uint80,
            int256 _price,
            uint256,
            uint256 updatedAt,
            uint80
        ) {
            price = uint256(_price);
            isStale = block.timestamp - updatedAt > 3600; // 1 hour staleness threshold
        } catch {
            price = 0;
            isStale = true;
        }
    }
    
    /**
     * @dev Calculate optimal arbitrage amount based on price impact
     */
    function calculateOptimalAmount(
        address tokenA,
        address tokenB,
        address dexA,
        address dexB
    ) external view returns (uint256 optimalAmount) {
        // This is a simplified calculation
        // In practice, you'd want to implement a more sophisticated algorithm
        // that considers liquidity depth and price impact curves
        
        uint256 baseAmount = 1000e18; // Start with 1000 tokens
        uint256 bestProfit = 0;
        optimalAmount = 0;
        
        for (uint256 i = 1; i <= 10; i++) {
            uint256 testAmount = baseAmount * i / 10;
            
            ArbitrageParams memory params = ArbitrageParams({
                tokenA: tokenA,
                tokenB: tokenB,
                dexA: dexA,
                dexB: dexB,
                minProfitBps: MIN_PROFIT_BPS,
                maxSlippageBps: MAX_SLIPPAGE,
                maxGasPrice: maxGasPrice
            });
            
            (bool profitable, uint256 profit,) = this.checkArbitrageOpportunity(params, testAmount);
            
            if (profitable && profit > bestProfit) {
                bestProfit = profit;
                optimalAmount = testAmount;
            }
        }
    }
    
    /**
     * @dev Update maximum gas price
     */
    function setMaxGasPrice(uint256 _maxGasPrice) external onlyOwner {
        uint256 oldPrice = maxGasPrice;
        maxGasPrice = _maxGasPrice;
        emit MaxGasPriceUpdated(oldPrice, _maxGasPrice);
    }
    
    /**
     * @dev Update profit share percentage
     */
    function setProfitShare(uint256 _profitShareBps) external onlyOwner {
        require(_profitShareBps <= 2000, "ArbitrageBot: Profit share too high"); // Max 20%
        profitShareBps = _profitShareBps;
    }
    
    /**
     * @dev Update treasury address
     */
    function setTreasury(address _treasury) external onlyOwner {
        require(_treasury != address(0), "ArbitrageBot: Invalid treasury");
        treasury = _treasury;
    }
    
    /**
     * @dev Withdraw accumulated profits (owner only)
     */
    function withdrawProfits(address token, uint256 amount) external onlyOwner {
        require(token != address(0), "ArbitrageBot: Invalid token");
        require(amount > 0, "ArbitrageBot: Invalid amount");
        
        IERC20(token).safeTransfer(owner(), amount);
        emit ProfitWithdrawn(owner(), amount);
    }
    
    /**
     * @dev Emergency withdrawal function
     */
    function emergencyWithdraw(address token) external onlyOwner {
        require(token != address(0), "ArbitrageBot: Invalid token");
        uint256 balance = IERC20(token).balanceOf(address(this));
        if (balance > 0) {
            IERC20(token).safeTransfer(owner(), balance);
        }
    }
    
    /**
     * @dev Pause contract operations
     */
    function pause() external onlyOwner {
        _pause();
    }
    
    /**
     * @dev Unpause contract operations
     */
    function unpause() external onlyOwner {
        _unpause();
    }
    
    /**
     * @dev Get contract statistics
     */
    function getStats() external view returns (
        uint256 _totalProfitGenerated,
        uint256 _totalArbitrageExecuted,
        uint256 _supportedDEXCount,
        address _priceOracle,
        uint256 _maxGasPrice
    ) {
        return (
            totalProfitGenerated,
            totalArbitrageExecuted,
            supportedDEXs.length,
            priceOracle,
            maxGasPrice
        );
    }
    
    /**
     * @dev Get DEX information
     */
    function getDEXInfo(address dex) external view returns (
        address router,
        uint256 fee,
        bool active,
        string memory name
    ) {
        DEXInfo memory info = dexInfo[dex];
        return (info.router, info.fee, info.active, info.name);
    }
    
    /**
     * @dev Get supported DEXs list
     */
    function getSupportedDEXs() external view returns (address[] memory) {
        return supportedDEXs;
    }
}